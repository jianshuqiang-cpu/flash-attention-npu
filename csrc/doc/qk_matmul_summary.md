# qk_matmul.hpp 详细解析

> **文件位置**: `csrc/flash_attn_npu/qk_matmul.hpp`
> **调度策略**: `MmadAtlasA2FAIQKT`（STAGES=2）
> **实例化位置**: `mha_fwd_kvcache.cpp`（L1TileShape=`<128,128,128>` 动态 N=128, L0TileShape=`<128,128,128>`）

---

## 1. 文件定位

本文件是 FlashAttention NPU 前向推理在 **Cube 核**上执行 **Q×K^T 矩阵乘** 的核心实现。它是 CATLASS 框架 `BlockMmad` 模板对 `MmadAtlasA2FAIQKT` 调度策略的特化。

在整个 FlashAttention 前向数据流中，QK 是**第一个**矩阵乘：

```
GM: Q, K, V
  │
  ▼ Cube核 (本文件)              ▼ Vector核
┌──────────┐                   ┌──────────────────┐
│ QK GEMM  │ S=Q×K^T ──GM───▶ │ online_softmax   │
│ (本文件) │                   │ (online_softmax. │
│          │ ──qkReady──▶      │  hpp)            │
└──────────┘                   └────────┬─────────┘
     │                                  │ P (softmax输出)
     │  qkReady=1                       │
     │  (CrossCoreSetFlag)              ▼
     │                            ┌──────────┐
     │                            │ PV GEMM  │ O_tmp = P×V → rescale_o → O
     │                            │(pv_matm  │
     │                            │ ul.hpp)  │
     │                            └──────────┘
```

QK 计算注意力原始分数 S，写入 workspace GM；随后通知 Vector 核开始 online softmax + causal mask。

---

## 2. Q×K^T 矩阵乘数学定义

给定当前 KV stack tile：

- Q ∈ ℝ^{M×K}：Query 矩阵（M = qSBlockSize × qNBlockSize，K = embed = d_head）
- K ∈ ℝ^{N×K}：Key 矩阵（N = stackSeqTile = 当前 KV 块长度，逻辑上需要转置）
- S ∈ ℝ^{M×N}：注意力分数矩阵（未加 mask、未 softmax）

$$S[m,n] = \sum_{k=0}^{K-1} Q[m,k] \cdot K[n,k]$$

注意：CATLASS 的 GEMM 接口约定 C = A × B，所以 K 矩阵在 DataCopy 时已按转置后的布局（B 矩阵）准备好，Cube 直接计算 Q × K^T 而无需显式转置操作。

---

## 3. QK 与 PV 的对偶性详解

QK 和 PV 虽然结构镜像，但关键区别在于**数据复用模式不同**：

| 维度 | QK (本文件) | PV (pv_matmul.hpp) |
|------|------------|-------------------|
| **A 矩阵角色** | Q（Query，固定不变） | P（Softmax输出，动态产出） |
| **B 矩阵角色** | K（Key，按KV块变化） | V（Value，固定不变） |
| **A 在 L1 的缓冲** | **单缓冲**（`l1ATensor`，loadQGM 一次加载） | **STAGES=2 Ping-Pong**（`l1ATensor[2]`，kL1循环内动态加载） |
| **B 在 L1 的缓冲** | **STAGES=2 Ping-Pong**（`l1BTensor[2]`，nL1循环内逐块加载） | **单缓冲**（`l1BTensor`，operator()开头一次性预加载） |
| **B 加载时机** | nL1 循环内（逐 KV 块加载） | operator() 开头（V 预加载，可与 softmax 并行） |
| **跨核信号方向** | CrossCore**Set**Flag（生产方，通知 Vector） | CrossCore**Wait**Flag（消费方，等 softmax） |
| **COORD_DIM1** | N = stackSeqTile（KV序列） | N = embed（embedding） |
| **COORD_DIM2** | K = embed（头维度） | K = stackSeqTile（KV序列） |
| **initMmad 条件** | `(kL0Idx == 0U)`（每个 mL0 块首 kL0 初始化） | `(kL1Idx==0)&&(kL0Idx==0)`（每个 mL1 块首子块初始化） |
| **外层循环顺序** | nL1(stack) → mL0(Q行) → kL0(embed) | nL1(embed) → mL1(Q行) → kL1(seq) → kL0(embed) |
| **kL1 外层循环** | **无**（embed=128 在一个 kL1 块内容纳） | **有**（KV seq 按 l1KDynamic=256 分块） |
| **GQA 支持** | loadQGM 专用多-pattern DataCopy | 无（P 矩阵已在 softmax 时处理好 GQA） |
| **Q/P 加载方式** | loadQGM() 独立接口，operator() 之前调用 | 在 kL1 循环内通过 copyGmToL1A 动态加载 |
| **排空信号** | EVENT_ID3 专用（Q 加载）、无入口排空 | EVENT_ID4 专用（PV 管道排空） |
| **L1 起始偏移** | 0（QK 先分配） | L1_QK_SIZE（紧接 QK 区域） |

**核心原因**：
- QK 阶段：Q 在整个 stack 计算中固定不变，适合单缓冲常驻；K 随 nL1 块切换，需要 Ping-Pong
- PV 阶段：V 在整个 stack 计算中固定不变，适合单缓冲预加载；P 由 Vector 核逐块产出，需要 Ping-Pong 动态加载

---

## 4. 内存层级与 L1 布局

```
GM (HBM)
├── gA = Q (loadQGM 预加载到 L1A)
├── gB = K (nL1 循环内 GM→L1B Ping-Pong)
└── gC = S (输出 workspace，供 online_softmax 消费)
         │
         │ MTE2 (GM→L1 DMA)
         ▼
L1 (片上 SRAM，从地址 0 开始)
┌─────────────────────────────────────┐ ← 0
│ L1A: Q 单缓冲                        │
│ 大小: M × kDyn × sizeof(ElementA)   │  如 128×128×2B = 32KB
├─────────────────────────────────────┤ ← M*kDyn*sizeof(A)
│ L1B[0]: K Ping-Pong slot 0          │
│ 大小: nDyn × kDyn × sizeof(ElementB)│  如 128×128×2B = 32KB
├─────────────────────────────────────┤
│ L1B[1]: K Ping-Pong slot 1          │
│ 大小: nDyn × kDyn × sizeof(ElementB)│  如 128×128×2B = 32KB
├─────────────────────────────────────┤ ← L1_QK_SIZE (PV 的 L1 从此开始)
│ PV 的 L1 (P Ping-Pong + V单缓冲)     │
└─────────────────────────────────────┘
         │
         │ MTE1 (L1→L0 DMA)
         ▼
L0 (Cube 单元缓冲)
├── l0ATensor[2]: Q 子块 (L0A Ping-Pong)
├── l0BTensor[2]: K 子块 (L0B Ping-Pong)
└── l0CTensor[2]: S 累加器 (L0C Ping-Pong)
         │
         │ Cube MMAD
         ▼
    L0C 累加 → FIX 管道 → GM (gC = S)
```

---

## 5. 类结构概览

```
BlockMmad<MmadAtlasA2FAIQKT<PAGED,ENABLE_UNIT>, L1<128,128,128>, L0<128,128,128>, ...>
│
├── 类型别名（同 PV，ElementA=Q, ElementB=K, ElementC=S）
├── 常量（STAGES=2, BLOCK_SIZE=16, COORD_DIM0/1/2）
│
├── 构造函数: 分配 L1/L0 缓冲
│   └── L1 布局: [Q单缓冲 | K Ping-Pong×2]
│
├── loadQGM(): **QK 特有** —— GQA-aware Q 矩阵预加载
│   └── 使用多-pattern DataCopy，参数含 tokenNumPerGroup/qHeads*embed
│
├── 分页辅助函数
│   ├── setBlockParam(): 计算页块参数（同 PV 逻辑）
│   ├── getBlockShape() ×2: 非分页/分页两种重载（QK特有：分页版在页内残留和L1容量间取min）
│   ├── getKVOffset() ×2: 非分页/分页偏移计算
│   ├── resetBlockStart(): 重置分页偏移
│   └── updateBlockOffset(): 更新块内偏移（QK版：仅页边界时curBlockIdx++）
│
├── operator(): 主入口
│   ├── 阶段0: 分页参数预计算
│   └── 阶段1-5: nL1→mL0→kL0 三层循环
│       ├── nL1 (KV序列分块, 步长l1NDynamic)
│       │   ├── [分页] while(kvL1Len < l1NResDynamic) 逐页拼接 K
│       │   └── [非分页] 单次 copyGmToL1B 加载 K
│       │   mL0 (Q行分块, 步长128)
│       │   ├── Wait FIX_M (等待C写回)
│       │   └── kL0 (embed分块, 步长128)
│       │       ├── L1A→L0A: Q子块
│       │       ├── L1B→L0B: K子块 (首块等K加载完成)
│       │       ├── 末mL0×末kL0: 释放L1B slot
│       │       ├── Set/Wait MTE1_M(EVENT_ID0)
│       │       ├── tileMmad(initMmad = (kL0Idx==0))
│       │       └── Set M_MTE1, flag^=1
│       │   Wait M_FIX → copyL0CToGm (S写回)
│       └── l1KvPingPongFlag ^= 1
│
└── 成员变量
    ├── l1ATensor (单缓冲), l1BTensor[2] (Ping-Pong)
    ├── l0ATensor[2], l0BTensor[2], l0CTensor[2]
    ├── copyGmToL1A (GQA多pattern), copyGmToL1B, copyL1ToL0A/B, copyL0CToGm
    ├── l1KvPingPongFlag (K), l0CPingPongFlag, l0ABPingPongFlag
    └── l1M/N/KDynamic, blockStartOffset, maxKVStackLen
```

---

## 6. loadQGM 的 GQA 支持详解

GQA (Grouped Query Attention) 中，多个 Q head 共享同一个 KV head。例如 `num_heads=32, num_kv_heads=8` 时，groupSize=4，每 4 个 Q head 共享 1 个 KV head。

Q 矩阵在 GM 中的物理布局（以 groupSize=4, qSBlockSize=128 tokens, 单group）：

```
GM 中 Q 的存储格式（interleaved）:
  token0_head0[128bf16] ┐
  token0_head1[128bf16] │ 一个 token 的所有 Q head 连续存储
  token0_head2[128bf16] │ 步长 = qHeads * embed 个元素
  token0_head3[128bf16] ┘
  token1_head0[128bf16]
  token1_head1[128bf16]
  ...
  token127_head3[128bf16]
```

loadQGM 的 DataCopy 参数：
- `layoutSingleANd = GetTileLayout([singleGroupHeads, embed])`：单次搬运 tile = qNBlockSize 个 head × embed
- `tokenNumPerGroup = rowNum / singleGroupHeads = qSBlockSize = 128`：每个 group 内有多少个 token
- GM 步长（源侧）：`qHeads * embed` —— 跳过其他 group 的 head 数据
- L1 目标步长：`tokenNumPerGroup = 128` —— 在 L1 中按 [rowNum, embed] 连续布局

经过 loadQGM 后，L1A 中的 Q 被整理为连续的 `[rowNum, embed]` 行主序布局，后续 L1→L0 搬运无需再感知 GQA。

---

## 7. 事件同步时序图

### 单个 nL1 块内的流水线时序

```
时间轴 ─────────────────────────────────────────────────▶

GM→L1(K): Wait [====copyGmToL1B(K_n)====] Set
L1→L0:            Wait   [Q L1→L0A]  Wait  [K L1→L0B]
Cube:                          Wait       [===MMAD(kL0=0)===]
                                       init=true,累加器初始化
L0C→GM:                                               Wait [===C→GM===]

第二kL0:                       Wait  [Q] Wait [K]
                                          [===MMAD(kL0=1)===]
                                       init=false,累加

第二mL0:            Wait ... Wait   Wait  Wait
                                   (重用L1B，因K已在L1)

事件:   WaitMTE1 WaitMTE2 SetMTE2  WaitM WaitM SetMTE1 WaitMTE1 WaitM_FIX
        _MTE2(0) _MTE1   _MTE1(0) _MTE1 _MTE1 _M(0)   _M(EV0)  ...
        (Kslot)  (首mL0) (K加载完) (L0A) (L0B) (L0A/B  (MMAD完)
                                       就绪)
```

### L1B Ping-Pong 机制

```
nL1Idx=0 (even): 使用 l1BTensor[0] 加载 K[0:128]
                 同时 nL1Idx=0 的 mL0/kL0 计算消耗 l1BTensor[0]

nL1Idx=1 (odd): 使用 l1BTensor[1] 加载 K[128:256]
                上一 nL1 的 mL0 循环可能仍在消耗 l1BTensor[0]，
                Wait MTE1_MTE2(1) 确保 l1BTensor[1] 已被消费完

释放时机：每个 nL1 块的最后一个 mL0 × 最后一个 kL0 完成后，
         Set MTE1_MTE2(l1KvPingPongFlag) 释放当前 L1B slot，
         供 nL1Idx+2 使用。
```

---

## 8. 例子 1：Prefill + Causal Mask + MHA（非分页）

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| seqlen_q | 512 | Prefill 序列长度 |
| seqlen_kv | 512 | KV 等长 |
| d_head (embed) | 128 | 头维度 |
| num_heads | 32 | MHA（Q头数=KV头数，无GQA） |
| qSBlockSize | 128 | 序列维分块大小 |
| qNBlockSize | 1 | MHA无GQA分组，单head处理 |
| rowNum | 128×1=128 | 当前task行数 |
| stackSeqTile | 256 | KV stack tile大小（非首块） |
| nDyn (l1NDynamic) | 128 | L1上K的N维动态tile |
| kDyn (l1KDynamic) | 128 | embed对齐后大小 |
| PAGED_CACHE_FLAG | false | 连续KV |
| 数据类型 | fp16 | 半精度 |
| L0TileShape | <128,128,128> | Cube基本块 |

### 循环次数

- `nL1Loop = CeilDiv(256, 128) = 2`（K 分2块：[0:128] 和 [128:256]）
- `mL0Loop = CeilDiv(128, 128) = 1`（Q 一整块 128 行）
- `kL0Loop = CeilDiv(128, 128) = 1`（embed=128，单kL0块）

### loadQGM 调用

```
rowNum=128, singleGroupHeads=qNBlockSize=1, qHeads=32
embed=128
rowNumRound=RoundUp(128, M_ALIGNED)=128
tokenNumPerGroup=128/1=128
layoutSingleANd: [1, 128]
单次DataCopy: 搬运 Q[0:128, 0:128] (fp16)，GM步长=32*128=4096
→ L1A 获得 Q[128×128]，EVENT_ID3 Set+Wait 确保完成
```

### operator() 执行追踪

```
nL1Idx=0, stackSeqTile=256, 非分页:
  getBlockShape(actualShape, 0, 2, 256): nActual=128 (非尾块)
  getKVOffset(gBOffset, nIdx=0, nowNIdx=0, strideKV=4096): gBOffset=0
  Wait MTE1_MTE2(0) → 通过
  copyGmToL1B(l1BTensor[0], K[0], [128,128]→[128,128]) → GM→L1B[0]
  Set MTE2_MTE1(0)

  mL0Idx=0, mL0Actual=128:
    Wait FIX_M(0) → 通过
    kL0Idx=0, kL0Actual=128:
      Wait M_MTE1(0) → 通过
      copyL1ToL0A(l0A[0], Q[0:128, 0:128]) → Q_00 到 L0A
      Wait M_MTE1(2) → 通过
      [mL0Idx==0 && kL0Idx==0] Wait MTE2_MTE1(0) → 等K加载完
      copyL1ToL0B(l0B[0], K[0:128, 0:128]) → K_00 到 L0B
      [mL0==last && kL0==last] Set MTE1_MTE2(0) → 释放L1B[0]
      Set/Wait MTE1_M(EVENT_ID0)
      initMmad=true (kL0Idx==0)
      mL0Align=128
      tileMmad(l0C[0], Q_00, K_00, 128, 128, 128, init=true)
        → L0C[0] = Q_00 × K_00
      Set M_MTE1(0), Set M_MTE1(2), l0ABFlag=1

    Wait M_FIX, copyL0CToGm(S[0:128, 0:128], l0C[0]) → S[0:128,0:128] 写回
    Set FIX_M(0), l0CFlag=1
  l1KvPingPongFlag=1

nL1Idx=1:
  getBlockShape(actualShape, 1, 2, 256): nActual=128 (尾块，恰好整除)
  getKVOffset: gBOffset=0 + 1*128*4096 = 524288 → K[128:256]
  Wait MTE1_MTE2(1) → 等L1B[1]空闲（初始通过）
  copyGmToL1B(l1BTensor[1], K[128], ...) → GM→L1B[1]
  Set MTE2_MTE1(1)
  mL0Idx=0, mL0Actual=128:
    Wait FIX_M(1) → 通过
    kL0Idx=0:
      Wait M_MTE1(1) → 通过
      copyL1ToL0A(l0A[1], Q[0:128,0:128]) → Q_00（Q常驻L1）
      Wait M_MTE1(3) → 通过
      [mL0==0 && kL0==0] Wait MTE2_MTE1(1) → 等K[128:256]加载
      copyL1ToL0B(l0B[1], K[128:256,0:128]) → K_10
      [mL0==last && kL0==last] Set MTE1_MTE2(1)
      Set/Wait MTE1_M(EVENT_ID0)
      initMmad=true
      tileMmad(l0C[1], Q_00, K_10, 128, 128, 128, init=true)
        → L0C[1] = Q_00 × K_10
      Set M_MTE1(1), Set M_MTE1(3), l0ABFlag=0
    Wait M_FIX → copyL0CToGm(S[0:128, 128:256])
    Set FIX_M(1), l0CFlag=0
  l1KvPingPongFlag=0
```

### S 输出布局

```
S [128 × 256] (fp16/float):
┌──────────────────┬──────────────────┐
│ S[0:128, 0:128]  │ S[0:128,128:256] │
│  = Q × K[0:128]^T│  = Q × K[128:256]^T
│  nL1Idx=0 写入    │  nL1Idx=1 写入    │
└──────────────────┴──────────────────┘
```

---

## 9. 例子 2：Decode + GQA(groupSize=8) + 分页 KV Cache + bf16

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| seqlen_q | 1 | Decode 单 token 生成 |
| seqlen_kv | 2048 | KV Cache 长度 |
| d_head (embed) | 128 | 头维度 |
| num_heads | 32 | Q头数 |
| num_kv_heads | 4 | KV头数（GQA groupSize=8） |
| qSBlockSize | 1 | 单token |
| qNBlockSize (curQNBlockTile) | 8 | 当前group内处理8个Q head |
| rowNum | 1×8=8 | 8行（同一KV group的8个Q head） |
| stackSeqTile | 512 | 每个PV调用处理512 KV token |
| nDyn (l1NDynamic) | 128 | L1上K的N维tile |
| blockSize | 128 | 分页块大小 |
| maxKVStackLen | 2048 | 最大KV长度 |
| PAGED_CACHE_FLAG | true | 分页KV |
| 数据类型 | bf16 | |

### loadQGM（GQA 路径）

```
rowNum=8, singleGroupHeads=qNBlockSize=8, qHeads=32
embed=128
tokenNumPerGroup=8/8=1 (每group 1个token)
layoutSingleANd: [8, 128] (8个Q head × 128 embed = 一个group)
GM步长 = qHeads*embed = 32*128 = 4096个bf16
DataCopy: 从Q[token0]处拷贝8个head×128个元素
→ L1A 获得 Q[8×128]，连续布局
```

Q 在 GM 中 GQA interleaved 布局：
```
GM 中 Q (seqlen=1, num_heads=32, embed=128):
token0: [head0(128bf16) head1 head2 head3 head4 head5 head6 head7 | head8-15 | head16-23 | head24-31]
        ↑ group0 (8 heads 共享 KV_head0)    ↑ group1  ↑ group2   ↑ group3

loadQGM 只搬运 group0 的8个head（head0-7），GM步长=4096跳过其他group
```

### operator() 分页执行追踪

nL1Loop = CeilDiv(512, 128) = 4，每个 nL1 块 128 tokens。

blockTable 前 4 个逻辑块（页）映射到物理块 [5,12,3,8]。

```
setBlockParam(512, blockStart=128, blockEnd=, curBlockTotalNum=, blockSize=128):
  stackSeqTile=512 >= blockStart=128
  blockEnd = (512-128)%128 = 384%128 = 0 → 三元运算: 整除→blockSize=128
  curBlockTotalNum = ceil(384/128)+1 = 3+1 = 4个物理块

nL1Idx=0, l1NResDynamic=128:
  Wait MTE1_MTE2(0)
  kvL1Len=0:
    curBlockSize=128 (curBlockIdx=0不是last)
    nowNIdx=0*16+0=0 (maxKVStackLen/blockSize=16)
    getBlockShape: nowLen = min(128-0, 128-0)=128
    getKVOffset: blockTable[0]=5, kOffset=5*128*strideKV+0
    copyGmToL1B(l1B[0][0:128,:], K物理页5[0:128,:])
    kvL1Len=128
    updateBlockOffset: 0+128==128 → blockStartOffset=0, curBlockIdx=1
  kvL1Len(128) >= l1NResDynamic(128) → 退出while
  Set MTE2_MTE1(0)
  nActual=128

  mL0Idx=0, mL0Actual=8 (rowNum=8 < 128):
    Wait FIX_M(0) → 通过
    kL0Idx=0, kL0Actual=128:
      Wait M_MTE1(0) → 通过
      copyL1ToL0A(l0A[0], Q[0:8, 0:128])
        l1ATileCoord = {0*128, 0*128} = {0,0}
      Wait M_MTE1(2) → 通过
      [mL0==0 && kL0==0] Wait MTE2_MTE1(0) → 等K加载完
      copyL1ToL0B(l0B[0], K_l1b0[0:128, 0:128])
      [mL0==last && kL0==last] Set MTE1_MTE2(0)
      Set/Wait MTE1_M(EVENT_ID0)
      initMmad=true
      mL0Align = (8+15)/16*16 = 16  ← 对齐到16行
      tileMmad(l0C[0], Q[0:8,:], K[0:128,:], 16, 128, 128, init=true)
        → L0C[0] = Q[0:8,:] × K[0:128,:]^T  → S[0:8, 0:128]（有效行8行，后8行padding）
      Set M_MTE1(0), Set M_MTE1(2), l0ABFlag=1
    Wait M_FIX → copyL0CToGm(S[0:8, 0:128])
    Set FIX_M(0), l0CFlag=1
  l1KvPingPongFlag=1

nL1Idx=1, l1NResDynamic=128:
  Wait MTE1_MTE2(1)
  kvL1Len=0:
    curBlockSize=128 (curBlockIdx=1 < 3)
    nowNIdx=0*16+1=1
    nowLen = min(128-0, 128-0)=128
    getKVOffset: blockTable[1]=12, kOffset=12*128*strideKV
    copyGmToL1B(l1B[1][0:128,:], K物理页12[0:128,:])
    kvL1Len=128
    updateBlockOffset: curBlockIdx=2
  Set MTE2_MTE1(1)
  mL0Idx=0, kL0Idx=0:
    (同nL1Idx=0流程，输出S[0:8, 128:256])
  l1KvPingPongFlag=0

nL1Idx=2: 物理页3 → S[0:8, 256:384] (Ping-Pong slot0复用)
nL1Idx=3: 物理页8 → S[0:8, 384:512] (Ping-Pong slot1复用)
```

### S 输出布局（Decode 单 token × 8 Q heads）

```
S [8 × 512] (bf16，分数矩阵):
┌─────┬──────┬──────┬──────┬──────┐
│head0│[0:128]│[128:256]│[256:384]│[384:512]│
│head1│  nL1  │  nL1  │  nL1  │  nL1  │
│...  │  =0   │  =1   │  =2   │  =3   │
│head7│      │      │      │      │
└─────┴──────┴──────┴──────┴──────┘
每个 nL1 块: 一次 MMAD，S 子块 [8×128]
后续 Vector 核 online_softmax 将对每一行独立计算 softmax。
```

### 分页 K 加载图示

```
GM 物理存储（blockSize=128 tokens/页）:
  ┌────┐  ┌────┐  ┌────┐  ┌────┐
  │页3 │  │页5 │  │页8 │  │页12│ ... （物理页）
  └────┘  └────┘  └────┘  └────┘

blockTable: [0]→5, [1]→12, [2]→3, [3]→8

nL1Idx=0 (l1B[0]): 从物理页5加载128tok → L1B[0][0:128]
nL1Idx=1 (l1B[1]): 从物理页12加载128tok → L1B[1][0:128]
nL1Idx=2 (l1B[0]): 从物理页3加载128tok → L1B[0][0:128]（slot0复用）
nL1Idx=3 (l1B[1]): 从物理页8加载128tok → L1B[1][0:128]（slot1复用）

这种情况下每个nL1块恰好对应一个完整物理页，无需跨页拼接。
若 blockSize≠nDyn 或边界有残留，while(kvL1Len<l1NResDynamic)会多次循环拼接。
```

---

## 10. 分页模式的跨页拼接逻辑详解

当 `blockSize=128, l1NDynamic=128` 时，每个 nL1 块恰好对应一个物理页，简单直接。但当参数不整除或有 blockStartOffset 残留时，需要跨页拼接：

**场景**：blockStartOffset=64（上次调用结束在物理页中间），stackSeqTile=200，blockSize=128，l1NDynamic=128

```
setBlockParam(200, blockStart=64, ...):
  stackSeqTile(200) >= blockStart(64)
  blockEnd = (200-64)%128 = 136%128 = 8 (末页取8 tokens)
  curBlockTotalNum = ceil(136/128)+1 = 2+1 = 3个物理块

nL1Idx=0 (l1NResDynamic=128):
  kvL1Len=0:
    curBlockIdx=0, curBlockSize=128 (非末块)
    nowLen = min(128-64, 128-0) = min(64, 128) = 64
    → 从当前物理页加载64 tokens到L1B[0:64]
    kvL1Len=64, blockStartOffset=64+64=128==blockSize→0, curBlockIdx=1
  kvL1Len=64:
    curBlockSize=128 (curBlockIdx=1非末块)
    nowLen = min(128-0, 128-64) = min(128, 64) = 64
    → 从新物理页加载64 tokens到L1B[64:128]
    kvL1Len=128, blockStartOffset=0+64=64
  → L1B[0:128] = [旧页尾64tok | 新页头64tok]，共128 tokens拼接完成
```

---

## 11. 设计亮点

1. **镜像 Ping-Pong 策略**：Q 常驻 L1 单缓冲（复用率最高），K 逐块 Ping-Pong 加载，与 PV 形成完美对偶。
2. **GQA 透明化**：loadQGM 使用多-pattern DataCopy 在 GM→L1 搬运时完成 GQA interleaved→连续布局的转换，后续计算路径无需感知 GQA。
3. **分页 KV 逐块拼接**：nL1 循环内 while 循环逐页加载 K 到 L1B，一个 L1B slot 可容纳来自多个物理页的拼接数据；通过 `nowLen = min(页内剩余, L1剩余)` 精确控制。
4. **L1 零开销共享**：QK 和 PV 共享同一块 L1 空间，QK 从地址0开始分配，PV 从 L1_QK_SIZE 偏移开始，两者不重叠也不浪费。
5. **三级独立 Ping-Pong**：L1B (K)、L0A/L0B (Q/K子块)、L0C (S输出) 三级独立双缓冲，通过 HardEvent 精细同步，最大化 DMA-Cube 并行。
6. **initMmad 精确控制**：`(kL0Idx==0)` 确保每个 mL0 块（每行块）的首个 kL0 子块初始化累加器，跨 embed 维累加正确。
7. **EVENT_ID3 专用同步**：loadQGM 使用 EVENT_ID3 而非 EVENT_ID0，避免与 operator() 内事件冲突。
8. **尾块对齐处理**：mL0Align 显式对齐到 BLOCK_SIZE=16；各维循环通过 `idx < Loop-1 ? TileSize : remainder` 处理非对齐尾块。

---

## 12. 注意事项

- `gA` 参数在 operator() 中传入但未使用（Q 已通过 loadQGM 预加载到 L1），仅用于接口一致性。
- `tileNNumPerBaseBlock` 变量已计算但未使用（预留用于优化）。
- `ENABLE_UNIT_FLAG_` 模板参数已声明但当前版本未使用。
- `EMBED_SPLIT_SIZE/UNIT_BLOCK_STACK_NUM/KV_BASE_BLOCK/KV_SPLIT_SIZE` 为预留常量，当前代码未引用。
- `l1MDynamic` 初始化为0但未赋值使用。
- QK 和 PV 的 `updateBlockOffset` 逻辑有细微差异：QK 仅在页边界时 curBlockIdx++，非页边界时累加偏移但不递增块索引（因为 K 的分页加载在 nL1 内可能跨多页）；PV 每次 updateBlockOffset 都 curBlockIdx++（因为 V 在 operator() 开头一次加载完所有页）。
- L1B 大小在构造时使用 `nDyn*kDyn` 动态计算，覆盖了模板静态 `L1B_SIZE = N*K*sizeof(B)` 的大小（N=128是模板默认值，nDyn运行时可能为128或更大受L1容量约束）。
- 分页模式下 `getBlockShape` 有两个重载，编译器通过参数个数/类型区分；分页模式使用5参数版本，非分页使用4参数版本。

---

## 13. 总结

`qk_matmul.hpp` 以约 340 行代码实现了 FlashAttention QK 阶段的高效 Cube GEMM，核心设计围绕以下几点：

1. **Q 常驻 + K Ping-Pong 的缓冲策略**：利用 Q 的跨 KV 块复用性最大化片上数据复用；
2. **loadQGM 的 GQA 透明加载**：在 GM→L1 阶段完成多头分组的数据重排，后续路径无 GQA 分支开销；
3. **分页 KV 在 nL1 循环内逐页拼接**：通过 while 循环 + min(页剩余, L1剩余) 的精确计算，将分散的物理页拼接为 L1 连续区域；
4. **多级 Ping-Pong 三级流水线**：L1B/L0AB/L0C 独立双缓冲 + HardEvent 同步，实现 DMA-Cube 高度并行；
5. **与 PV 的镜像对偶设计**：两者共享 L1、共享大部分同步模式，仅在缓冲策略和外层循环顺序上镜像对称。

QK 输出的 S 矩阵经 Vector 核 online_softmax 归一化（加 causal mask）后得到 P，再送入 pv_matmul.hpp 完成 P×V 矩阵乘，最终由 rescale_o.hpp 完成跨-stack 累加和归一化，得到完整的 FlashAttention 输出 O。
