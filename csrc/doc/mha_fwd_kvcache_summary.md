# mha_fwd_kvcache.cpp 详解

## 1. 文件定位

[mha_fwd_kvcache.cpp](../flash_attn_npu/mha_fwd_kvcache.cpp) 是 FlashAttention NPU **前向推理** kernel 的核心实现文件，位于：

```
csrc/flash_attn_npu/mha_fwd_kvcache.cpp
```

它是整个前向 pipeline 的"总指挥"文件，将以下组件组装成一个完整的 FlashAttention kernel：

| 组件 | 文件 | 职责 | 运行位置 |
|------|------|------|---------|
| Q*K^T 矩阵乘 | [qk_matmul.hpp](../flash_attn_npu/qk_matmul.hpp) | 计算 attention scores S | Cube 核 |
| Online Softmax | [online_softmax.hpp](../flash_attn_npu/online_softmax.hpp) | scale/mask/exp/rowmax/rowsum | Vector 核 |
| P*V 矩阵乘 | [pv_matmul.hpp](../flash_attn_npu/pv_matmul.hpp) | 计算 attention 加权值 | Cube 核 |
| Rescale O | [rescale_o.hpp](../flash_attn_npu/rescale_o.hpp) | O 重缩放/累加/归一化 | Vector 核 |
| Init Outputs | [init_outputs.hpp](../flash_attn_npu/init_outputs.hpp) | 零 KV 边界 O/LSE 初始化 | Vector 核 |
| Tiling 数据 | [tilingdata.h](../flash_attn_npu/tilingdata.h) | Host→Device 标量参数 | — |
| 公共定义 | [kernel_common.hpp](../flash_attn_npu/kernel_common.hpp) | 常量/枚举/参数结构体 | — |

---

## 2. 架构核心：Cube + Vector 双异构核协作

Atlas A2 (C220) 架构上每个 AI Core 包含：
- **1 个 Cube 计算单元**：执行矩阵乘（MMAD）
- **2 个 Vector 子核**（sub-core）：执行逐元素运算

编译器对同一个 kernel 入口**编译两次**，通过 `__DAV_C220_CUBE__` 和 `__DAV_C220_VEC__` 宏选择代码路径。两侧通过 `CrossCoreFlag` 做核间生产者-消费者同步：

```
           同一个 AI Core
┌─────────────────────────────────────┐
│  Cube 核（1个）    │  Vector 子核（2个）│
│                   │                   │
│  Q*K^T ──flag1──► │  softmax+mask     │
│                   │       │           │
│  P*V   ◄──flag2── │       │           │
│    │              │       ▼           │
│    └────flag3───► │  rescale O 累加   │
└─────────────────────────────────────┘
     共享 GM 地址空间 + CrossCoreFlag 信号量
```

### 双编译模式的代码可见性

- `#ifdef __DAV_C220_CUBE__` 包围的代码：只在 Cube 核上编译执行（BlockMmadQK/BlockMmadPV 构造和调用、Q 加载、L1 tile 计算、Cube 侧 SetFlag/WaitFlag）
- `#ifdef __DAV_C220_VEC__` 包围的代码：只在 Vector 核上编译执行（三个 Epilogue 构造和调用、Vector 侧 SetFlag/WaitFlag）
- 宏外的代码：两侧共享（tiling 解析、GlobalTensor 绑定、stride 计算、task 循环控制、GM offset 计算）

Vector 侧 `coreIdx = GetBlockIdx() / GetSubBlockNum()` 将 sub-core 编号除以 2 映射到 AI Core 逻辑编号，与 Cube 侧对齐，保证同一核上两侧共享 workspace。

---

## 3. 文件结构

文件包含两个 `namespace SplitFuse` 作用域：

```
mha_fwd_kvcache.cpp
├── namespace SplitFuse { ... }           (L77-L721)
│   └── class FAInferKernel<...>          ← 主 kernel 类
│       ├── using 类型别名                 ← 从模板参数提取所有类型
│       ├── operator()(params)            ← kernel 主函数
│       │   ├── ① tiling 解析 + GlobalTensor 绑定
│       │   ├── ② 灌泡 SetFlag（Cube/Vector 两侧）
│       │   ├── ③ L1 tile 计算 + GEMM/Epilogue 对象构造
│       │   ├── ④ stride/对齐/groupSize 计算
│       │   ├── ⑤ TND 布局变长任务数重算
│       │   ├── ⑥ task 循环（多核分配）
│       │   │   ├── while: 跨 batch 推进
│       │   │   ├── taskIdx → 坐标解码
│       │   │   ├── 零 KV 边界 InitOut
│       │   │   ├── loadQGM（Q 一次性加载）
│       │   │   └── KV 外层循环（PRE_LAUNCH=2 三缓冲流水）
│       │   │       ├── QK matmul (Cube)
│       │   │       ├── Online Softmax (Vector)
│       │   │       ├── PV matmul (Cube, 延迟2轮)
│       │   │       └── Rescale O (Vector, 延迟2轮)
│       │   ├── ⑦ 排空 WaitFlag（Cube/Vector 两侧）
│       │   └── ⑧ PipeBarrier<PIPE_ALL>()
│   └── private 成员: resource + 3 个 CrossCoreFlag
│
└── namespace SplitFuse { ... }           (L723-L837)
    └── __global__ void FAInfer<...>(...) ← 全局 kernel 入口
        ├── SetSyncBaseAddr(fftsAddr)     ← 设置跨核同步基地址
        ├── using 声明：组装所有模板类型
        ├── 构造 BlockMmadQK/PV/Epilogue 类型
        ├── FAIKernelParams 打包
        └── FAInferKernel{}.operator()(params)
```

---

## 4. 核心流程详解

### 4.1 Tiling 解析与 GlobalTensor 绑定

从 GM 上的 `FAInferTilingData` 读取标量参数（batch/heads/embed/scale/maskType 等），然后将所有张量指针绑定到 `GlobalTensor`：

```cpp
__gm__ FAInferTilingData *tiling = (__gm__ FAInferTilingData*)params.tiling;
// 读取 batch/qHeads/kvHeads/embed/embedV/scaleValue/softcapValue/...

gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
// ... gV/gMask/gBlockTable/gActualQseqlen/gActualKvseqlen/gO/gLse

// workspace 切四区域
gS       = params.workSpace;                                    // QK scores
gP       = params.workSpace + mm1OutSize;                       // softmax probs
gOTmp    = params.workSpace + mm1OutSize + smOnlineOutSize;     // PV 输出
gOUpdate = params.workSpace + mm1OutSize + smOnlineOutSize + mm2OutSize; // online softmax 状态
```

### 4.2 灌泡（SetFlag）与排空（WaitFlag）

开头的一堆 `SetFlag` 是**流水线灌泡**：双缓冲/多缓冲流水线中第一轮没有"上一轮"释放 buffer，因此手动将所有 event 置为 signaled，相当于"所有 buffer 初始空闲"。

结尾对应的一堆 `WaitFlag` 是**流水线排空**：循环结束后仍有 in-flight DMA 和计算，等待所有操作完成确保数据写回 GM。

```
                   时间 →
开头 SetFlag: ●●●●●●●●  (所有 buffer 标为空闲)
              │
KV 循环:     [QK][SM][PV][RO]
                 [QK][SM][PV][RO]
                     [QK][SM][PV][RO]  ← 稳态
                          ...
                                  [QK][SM][PV][RO]
结尾 WaitFlag:                    WWWWWWWW  (等待所有完成)
```

### 4.3 L1 Tile 动态计算（Cube 侧）

```
L1 总 512KB
├── V 预留: embedV × 512 × sizeof(half) × 2(双缓冲) = 256KB (embed=128 时)
└── Q+K 剩余: ~256KB
    ├── Q 固定: 128 × kDynNum × sizeof(half)
    └── K 动态: 剩余空间 → nDynNum ≤ min(预算, 128), 对齐到 32
```

- `kDynNum = RoundUp(embed, 128)`，最小 256
- `nDynNum = L1 剩余空间反推 K 列方向 tile 大小`，不超过 128

### 4.4 多核任务分配

```cpp
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum)
```

每个核按 `coreIdx` 为起点、`coreNum` 为步长分配任务，实现均匀负载。taskIdx 到具体 (qSBlockIdx, qNBlockIdx) 的解码：

```cpp
taskIdxCurBatch = taskIdx - preTotalTaskNum;
qSBlockIdx = taskIdxCurBatch / curQNBlockNum;          // 序列维块
qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum; // head 维块
kvNIdx     = qNBlockIdx / qNBlockNumPerGroup;          // KV head 索引
qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile; // Q head 起始
```

跨 batch 时通过 `while (taskIdx >= curTotalTaskNum)` 推进 curBatch，累加 GM offset。

### 4.5 因果 Mask 的 noSkipKvS 计算

```cpp
int64_t noSkipKvS = kvSeqlen;                      // 非 causal: 全部计算
if (maskType != 0) {
    diffS = max(kvSeqlen - qSeqlen, 0);           // KV 前缀长度
    noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*128 + diffS);
}
kvSLoopNumTotal = CeilDiv(noSkipKvS, 512);
```

因果 mask 下，当前 Q block 不需要关注"未来"的 KV token，因此可以跳过那部分，减少 KV 外层循环次数。Causal fast path：完全在对角线下方的 KV stack 块（`kvSEndIdx <= triUp+1`）不需要 mask，走快速路径。

### 4.6 KV 外层循环（PRE_LAUNCH=2 三缓冲深度流水）

实际循环次数 = `kvSLoopNumTotal + 2`（额外 2 轮用于排空）：

- **发射阶段**（前 `preKVNum=2` 轮）：只做 QK+Softmax，填满流水线
- **稳态阶段**：QK+Softmax+PV+RescaleO 同时进行
- **排空阶段**（最后 `preKVNum=2` 轮）：只做 PV+RescaleO，排空流水线

三缓冲槽位：`curStackTileMod = stackSeqCount % 3`，PV 消费落后 2 轮：`(stackSeqCount - PRE_LAUNCH) % 3`。

### 4.7 Online Softmax 块内逻辑

每个 KV stack 块内，online softmax 维护：
- `m`（rowmax：当前行最大值）
- `l`（rowsum：exp 校正后的和）
- `o`（输出 O 累加值）

分首块/中间块/末块：
- **首块**（`stackSeqCount == 0`）：`m = rowmax(S)`, `l = rowsum(exp(S-m))`, `o = P*V`
- **中间块**：计算新 `m_new`, `l_new`，`o = o*exp(m-m_new) + P*V`, `l = l*exp(m-m_new) + l_new`
- **末块**：最后归一化 `o = o / l`

---

## 5. 例子 1：Prefill 长序列（BSND + MHA + Causal）

**配置**：batch=2, qSeqlen=1024, kvSeqlen=1024, qHeads=32, kvHeads=32(MHA), embed=128, causal, 8 个 AI Core

**Tiling 结果**：
- `qSBlockTile=128`，`qSBlockNum=8`（序列方向 8 块）
- `qNBlockTile=1`（MHA 每块 1 个 Q head），`qNBlockNum=32`
- 总 task 数 = 2 × 8 × 32 = 512
- 每核处理 512/8 = 64 个 task

**KV 循环**：
- 对第一个 Q block (qSBlockIdx=0)，`noSkipKvS = 128 + 0 = 128`（causal 只算前 128 个 KV），`kvSLoopNumTotal=1`
- 对中间 Q block (qSBlockIdx=3)，`noSkipKvS = 512`，`kvSLoopNumTotal=1`（完全在对角线下，无 mask fast path）
- 对最后 Q block (qSBlockIdx=7)，`noSkipKvS = 1024`，`kvSLoopNumTotal=2`

**流水时序**（以 qSBlockIdx=7 为例，kvSLoopNumTotal=2，总循环=4 轮）：

```
kvSIdx:      0       1       2       3
Cube-QK:   [QK0]   [QK1]
Vec-SM:        [SM0]   [SM1]
Cube-PV:            [PV0]   [PV1]
Vec-RO:                 [RO0]   [RO1]
           发射  ← 稳态 →  排空
```

**GM 数据流**：

```
GM Q[B,1024,32,128]  ──loadQGM(一次性)──► L1
GM K[B,1024,32,128]  ──逐KV块DMA───────► L1
GM V[B,1024,32,128]  ──逐KV块DMA───────► L1
                                          │
         ┌──── Q*K^T (Cube) ────► gS(workspace, float)
         │                              │
         ▼                      softmax (Vector) ──► gP(workspace, half)
     gS(workspace)                        │
         │                                ▼
         │                         P*V (Cube) ──► gOTmp(workspace, float)
         │                                │
         │                                ▼
         └────────────────── rescale O (Vector) ──► GM O[B,1024,32,128]
                                                   GM LSE[B,1024,32]
```

---

## 6. 例子 2：Decode 单 Token（TND + GQA + Paged KV Cache + 无 mask）

**配置**：batch=64(packed), 每请求 qSeqlen=1(decode), kvSeqlen≈2048, qHeads=64, kvHeads=8(GQA 8:1), embed=128, paged KV cache(blockSize=128), 8 个 AI Core

**Tiling 结果**：
- `qSBlockTile=128`，但 qSeqlen=1 所以 `qSBlockNum=1`（单块，尾块大小=1）
- `qNBlockTile=GetQNBlockTile(1, 8)=8`（一次处理 1 个 KV group 的全部 8 个 Q head）
- `qNBlockNum = CeilDiv(8,8)*8 = 8`
- 总 task 数 = 64 × 1 × 8 = 512
- 每核处理 64 个 task

**KV 循环**：
- 非 causal，`noSkipKvS=kvSeqlen≈2048`，`kvSLoopNumTotal = CeilDiv(2048,512)=4`
- 总循环 = 4+2 = 6 轮
- paged 模式下，blockMmadQK/blockMmadPV 通过 `gBlockTable` 页表查找物理页地址

**流水时序**（kvSLoopNumTotal=4）：

```
kvSIdx:      0     1     2     3     4     5
Cube-QK:   [QK0][QK1][QK2][QK3]
Vec-SM:       [SM0][SM1][SM2][SM3]
Cube-PV:           [PV0][PV1][PV2][PV3]
Vec-RO:               [RO0][RO1][RO2][RO3]
             \发射/  \    稳态     /  \排空/
```

**Paged KV Cache 寻址**：

```
blockTables (GM, int32): 每个 batch 一个 page 索引序列
    batch0: [page3, page7, page12, page1, ...]  ← 物理页号
    batch1: [page5, page9, page0, ...]
    ...

K/V 实际地址 = pageBaseAddr + blockTable[batchOffset + pageIdx] * blockSize * strideKV
每个 MAX_KV_STACK_LEN=512 个 KV token 跨 ceil(512/128)=4 个 page
```

**GQA head 处理**：
- qNBlockTile=8 表示一次处理 8 个 Q head（属于同一个 KV head group）
- 每个 KV head (kvNIdx=0..7) 对应 8 个 Q head
- Q 在 head 维的起始索引 = `kvNIdx * 8 + qNBlockIdxCurGroup * 8`

---

## 7. 数据流/结构图

### 7.1 整体 Kernel 架构（Cube/Vector 双编译视角）

```
                    __global__ void FAInfer(...)
                           │
              SetSyncBaseAddr(fftsAddr)
              类型组装 (using 声明)
              FAIKernelParams 打包
                           │
                           ▼
              FAInferKernel::operator()(params)
                    ┌──────┴──────┐
                    │ 双编译分叉   │
          ┌─────────┴─────────────┴─────────┐
          │                               │
    __DAV_C220_CUBE__               __DAV_C220_VEC__
    (Cube 核路径)                   (Vector 子核×2 路径)
          │                               │
    ┌─────┴─────┐                  ┌──────┴──────┐
    │SetFlag灌泡│                  │SetFlag灌泡  │
    │L1 tile计  │                  │Epilogue构造 │
    │算+GEMM构造│                  │coreIdx对齐   │
    └─────┬─────┘                  └──────┬──────┘
          │     共享 task 循环逻辑          │
          └────────────┬──────────────────┘
                       ▼
              for taskIdx (多核分配)
                ┌──────┴──────┐
                │ while跨batch │
                │ 坐标解码     │
                │ GM offset   │
                └──────┬──────┘
           ┌───────────┴───────────┐
           │ Cube: loadQGM(一次性) │ Vec: InitOut(零KV)
           └───────────┬───────────┘
                       ▼
              for kvSIdx (KV外层循环, 三缓冲流水)
           ┌───────────────────────────────┐
           │ if kvSIdx < kvSLoopNumTotal: │
           │  Cube: blockMmadQK() ──flag1►│
           │  Vec:  WaitFlag1             │
           │       epilogueOnlineSoftmax()│
           │                      ◄──flag2│
           │ if kvSIdx >= preKVNum(=2):   │
           │  Cube: WaitFlag2             │
           │       blockMmadPV() ──flag3► │
           │  Vec:  WaitFlag3             │
           │       epilogueRescaleO()     │
           └───────────────────────────────┘
                       │
              ┌────────┴────────┐
              │WaitFlag排空(Cube)│WaitFlag排空(Vector)
              └────────┬────────┘
                       ▼
              PipeBarrier<PIPE_ALL>()
```

### 7.2 Workspace 布局（单核）

```
GM workSpace 指针
  │
  ▼
┌────────────────────────┬─────────────────────┬─────────────────────┬──────────────────────┐
│  gS (QK scores)        │  gP (softmax probs)  │  gOTmp (PV output)  │  gOUpdate (m/l/dm)   │
│  128×512 × sizeof(S)  │  128×512 × half     │  128×128 × float    │  online softmax 状态  │
│  三缓冲 (×3)           │  三缓冲 (×3)         │  三缓冲 (×3)         │  单缓冲              │
│  偏移 mod%3            │  同 gS 偏移           │  同 gS 偏移          │  固定偏移             │
└────────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘
  ▲
  │ coreIdx * WORKSPACE_BLOCK_SIZE_DB * 3  （每核独立 workspace）
```

workspace 总大小 = `blockDim × (128*512) × sizeof(float) × 3 × 4`。

### 7.3 L1 内存布局（Cube 核）

```
┌──────────────────── L1: 512 KB ────────────────────┐
│                                                      │
│  ┌── V 双缓冲 ──┐  ┌─ Q (固定) ─┐  ┌── K 动态 ────┐ │
│  │ ping: embedV │  │ 128×kDynNum│  │ nDynNum ×    │ │
│  │   ×512×2B   │  │ ×2B        │  │ kDynNum×2B   │ │
│  │ pong: embedV│  │ (约64KB)   │  │ ×2(双缓冲)    │ │
│  │   ×512×2B   │  │            │  │ (约 128KB)    │ │
│  │ = 256KB     │  │            │  │               │ │
│  └─────────────┘  └────────────┘  └───────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 7.4 Task 二维分块示意（BSND 布局 Q 矩阵）

```
Q 矩阵 [B, S=qSeqlen, N=qHeads, D=embed]
每个 task 处理一个 (batch, sBlock, nBlock)

                 head 维 N
        ◄──── qNBlockTile heads ────►
        ┌────┬────┬────┬────┬────┬───┐
     ┌──┤B0S0│B0S0│B0S0│...         │  qSBlock0 (128 行)
     │  │ N0 │ N1 │ N2 │            │
  S  │  ├────┼────┼────┼────┼────┼───┤
  维 │  │B0S1│B0S1│...              │  qSBlock1 (128 行)
     │  │ N0 │ N1 │                 │
     │  ├────┼────┼────┼────┼────┼───┤
     │  │ ...                        │
     ▼  ├────┼────┼────┼────┼────┼───┤
        │B1S0│B1S0│...              │  下一个 batch
        │ N0 │ N1 │                 │
        └────┴────┴────┴────┴────┴───┘
        每个小格 = 1 个 task（qSBlockSize × qNBlockSize 个"行"）
        8 个核按 taskIdx 交错分配：core0→task0,8,16...; core1→task1,9,17...
```

### 7.5 Causal Mask KV 跳过逻辑

```
             KV 序列位置 →
          0   128  256  384  512  640  768  896  1024
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
Q 0-127 │ ✓✓✓│MASK│MASK│MASK│MASK│MASK│MASK│MASK│ qSBlock0: noSkip=128, 1个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
128-255 │ ✓ │ ✓✓✓│MASK│MASK│MASK│MASK│MASK│MASK│ qSBlock1: noSkip=256, 1个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
256-383 │ ✓ │ ✓  │ ✓✓✓│MASK│MASK│MASK│MASK│MASK│ qSBlock2: noSkip=384, 1个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
384-511 │ ✓ │ ✓  │ ✓  │ ✓✓✓│MASK│MASK│MASK│MASK│ qSBlock3: noSkip=512, 1个KV块, 全对角线下fast path
        ├────┼────┼────┼────┼────┼────┼────┼────┤
512-639 │ ✓ │ ✓  │ ✓  │ ✓  │ ✓✓✓│MASK│MASK│MASK│ qSBlock4: noSkip=640, 2个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
640-767 │ ✓ │ ✓  │ ✓  │ ✓  │ ✓  │ ✓✓✓│MASK│MASK│ qSBlock5: noSkip=768, 2个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
768-895 │ ✓ │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │ ✓✓✓│MASK│ qSBlock6: noSkip=896, 2个KV块
        ├────┼────┼────┼────┼────┼────┼────┼────┤
896-1023│ ✓ │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │ ✓  │✓✓✓│ qSBlock7: noSkip=1024, 2个KV块
        └────┴────┴────┴────┴────┴────┴────┴────┘
✓ = 有效（需要计算）  MASK = 被 causal mask 跳过（不进入 KV 循环）
✓✓✓ = 跨越对角线的 KV 块（需要从预生成 mask 中切子窗应用）
```

---

## 8. Kernel 模板实例化（flash_api.cpp 侧）

Host 侧 `flash_api.cpp` 根据实际配置选择模板实例：

| 模板参数 | 可选值 | 说明 |
|---------|-------|------|
| `InputDtypeQ` | half / bfloat16_t | Q 数据类型 |
| `InputDtypeKv` | half / bfloat16_t | K/V 数据类型 |
| `IntermCalcPrec` | float (当前全部使用) | 中间精度（half 低精度预留） |
| `PagedCacheFlag` | true / false | 是否 paged KV cache |
| `maskCategory` | NO_MASK / MASK_CAUSAL | mask 类型 |
| `inLayout` | BSND / TND | 输入布局 |
| `lseMode` | NONE / OUT_ONLY | 是否输出 LSE |

主要入口函数：
- `mha_fwd_kvcache`：KV-cache 推理（支持 paged）
- `mha_fwd`：标准 BSND 前向
- `mha_varlen_fwd`：变长（TND packed）前向

最多实例化 `2×2×2×2 = 16` 种组合。

---

## 9. 注意点

1. **Q 只加载一次**：`loadQGM` 在 KV 循环外调用，Q 矩阵在 L1 中驻留整个 KV 循环过程（K/V 逐块加载）。这是 FlashAttention 的关键优化。
2. **三缓冲深度 = PRE_LAUNCH+1 = 3**：workspace 偏移按 `%3` 计算，修改 `PRE_LAUNCH` 需同步修改所有缓冲槽位逻辑。
3. **Vector sub-core 对齐**：Vector 侧 `coreIdx = GetBlockIdx()/GetSubBlockNum()` 保证与 Cube 侧 coreIdx 一致，共享 workspace。
4. **尾块处理**：最后一个 qSBlock/qNBlock/kvSBlock 的大小可能不足 tile 大小，通过 if 判断实际大小。
5. **Paged vs 连续 KV**：通过 `if constexpr (PAGED_CACHE_FLAG)` 编译期裁剪，paged 模式下 K/V 地址通过 blockTable 页表查找。
6. **TND vs BSND**：TND 布局下在 device 侧通过 cumulative seqlen 前缀差分重新计算每 batch 长度。
7. **noSkipKvS 因果优化**：causal 下跳过未来 KV token 的计算，减少外层 KV 循环次数，显著提升 prefill 性能。
8. **首块/末块标记**：`stackSeqCount==0` 标记首块（初始化 m/l），`nowkvSIdx+1 >= kvSLoopNumTotal` 标记末块（最终归一化）。
9. **CrossCoreFlag 依赖 fftsAddr**：kernel 入口必须调用 `SetSyncBaseAddr(fftsAddr)` 设置跨核信号量共享地址。
10. **gmOffsetP = gmOffsetS**：softmax 输出 P 覆盖 S 的位置（in-place），节省 workspace 空间。

---

## 10. 总结

[mha_fwd_kvcache.cpp](../flash_attn_npu/mha_fwd_kvcache.cpp) 是 FlashAttention NPU 前向推理的**核心调度文件**，它：

1. **双编译实现异构并行**：通过 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 宏让 Cube 核执行矩阵乘、Vector 核执行 softmax/epilogue，物理并行；
2. **三段 CrossCoreFlag 同步**：qkReady/softmaxReady/pvReady 三个信号实现 QK→Softmax→PV→RescaleO 的深度流水；
3. **PRE_LAUNCH=2 三缓冲**：3 个 workspace 槽位让生产-消费错位 2 轮，填满三段流水线达到稳态吞吐；
4. **在线 softmax 算法**：维护 m/l/o 三元组，在 KV 循环内逐块更新，不需要物化完整的 S=QK^T 矩阵；
5. **多维度编译期裁剪**：PagedCacheFlag/MASK_TYPE/INPUT_LAYOUT 三个 bool/enum 模板参数通过 `if constexpr` 编译期裁剪无用分支；
6. **因果 mask 智能跳过**：noSkipKvS 计算减少不必要的 KV 循环；完全在对角线下的块走无 mask 快速路径；
7. **多核任务分发**：按 coreIdx 步长均匀分配 task，支持 TND 变长布局下的动态任务数计算。

理解这个文件就理解了 FlashAttention NPU 前向 kernel 的完整执行流程。它的其他辅助文件（qk_matmul.hpp、online_softmax.hpp、pv_matmul.hpp、rescale_o.hpp、init_outputs.hpp）都是这个总指挥下的"专职工人"，分别负责流水线中的一个具体阶段。
