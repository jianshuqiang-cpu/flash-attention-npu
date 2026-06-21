# pv_matmul.hpp 详细解析

> **文件位置**: `csrc/flash_attn_npu/pv_matmul.hpp`
> **调度策略**: `MmadAtlasA2FAIPVT`（STAGES=2）
> **实例化位置**: `mha_fwd_kvcache.cpp`（L1TileShape=`<128,128,256>`, L0TileShape=`<128,128,128>`）

---

## 1. 文件定位

本文件是 FlashAttention NPU 前向推理在 **Cube 核**上执行 **P×V 矩阵乘** 的核心实现。它是 CATLASS 框架 `BlockMmad` 模板对 `MmadAtlasA2FAIPVT` 调度策略的特化。

在整个 FlashAttention 前向数据流中：

```
GM: Q, K, V
  │
  ▼ Cube核                     ▼ Vector核
┌──────────┐                ┌──────────────────┐
│ QK GEMM  │ S=Q×K^T ──GM──▶│ online_softmax   │
│ (qk_matm │                │ (online_softmax. │
│  ul.hpp) │                │  hpp)            │
└──────────┘                └────────┬─────────┘
     ▲                              │ P (softmax输出)
     │                              │ workspace
     │                              ▼
     │  softmaxReady          ┌──────────┐
     └────────────────────────│ PV GEMM  │ O_tmp = P×V ──GM──▶ rescale_o.hpp
         CrossCoreWaitFlag    │(本文件)  │                    累加+归一化 → O
                              └──────────┘
                                   │ pvReady
                                   ▼
                              Vector核 rescale_o
```

PV 是 FlashAttention 的第二个矩阵乘，位于 softmax epilogue 之后，将归一化的注意力权重 P 与 Value 矩阵 V 相乘得到当前 KV stack 的输出贡献 OTmp。

---

## 2. P×V 矩阵乘数学定义

给定当前 KV stack tile：

- P ∈ ℝ^{M×K}：注意力权重矩阵（M = qBlockSize = Q行数，K = stackSeqTile = KV序列长度）
- V ∈ ℝ^{K×N}：Value 矩阵（N = embed = d_head = 128）
- OTmp ∈ ℝ^{M×N}：输出累加器

$$O_{tmp}[m,n] = \sum_{k=0}^{K-1} P[m,k] \cdot V[k,n]$$

在多 KV stack 场景下，每个 stack 计算出部分 OTmp，由下游 [rescale_o.hpp](rescale_o.hpp) 做：
- O = O + OTmp（跨 stack 累加）
- 最终 O = O / L（除以 softmax 归一化因子），其中 L 来自 online softmax 的 l 累加器。

---

## 3. QK 与 PV 的对偶性

QK（S=Q×K^T）和 PV（O=P×V）虽然都是 GEMM，但由于数据复用模式不同，采用了**镜像对称**的 Ping-Pong 策略：

| 维度 | QK (qk_matmul.hpp) | PV (本文件) |
|------|-------------------|-------------|
| **A 矩阵** | Q：单缓冲常驻 L1 | P：STAGES=2 Ping-Pong 动态加载 |
| **B 矩阵** | K：STAGES=2 Ping-Pong 动态加载 | V：单缓冲预加载到 L1 |
| **A 加载时机** | 循环外一次 `loadQGM()` | kL1 循环内 `copyGmToL1A` |
| **B 加载时机** | nL1 循环开始时 | **operator() 开头一次性预加载** |
| **跨核等待** | 无 | `CrossCoreWaitFlag(softmaxFlag)` |
| **COORD_DIM1** | N = stackSeqTile (K维序列) | N = embed |
| **COORD_DIM2** | K = embed | K = stackSeqTile |
| **L1 内存布局** | [Q单缓冲 \| K PingPong×2] | [P PingPong×2 \| V单缓冲] |
| **initMmad 条件** | `(kL0Idx == 0U)` | `(kL1Idx==0) && (kL0Idx==0)` |
| **外层循环顺序** | nL1(stack)→mL0→kL0 | nL1(embed)→mL1→kL1→kL0 |

**核心原因**：QK 阶段 Q 矩阵是固定的（同一批 query 对多个 KV 块复用），而 K 随 nL1 变化；PV 阶段 V 矩阵在整个 stack 计算中固定，而 P 是 softmax 刚产出的新数据，需要动态加载。

---

## 4. 内存层级

```
┌──────────────────────────────────────────────────────────────┐
│ GM (全局内存, HBM)                                          │
│  - gA = P 矩阵 (softmax 输出 workspace)                     │
│  - gB = V 矩阵                                              │
│  - gC = OTmp 输出                                           │
│  - gBlockTable = 分页块表                                    │
└──────────────┬───────────────────────────────────────────────┘
               │ MTE2 (GM↔L1 DMA)
               ▼
┌──────────────────────────────────────────────────────────────┐
│ L1 (片上 SRAM, 与 QK 共享, PV 从偏移 L1_QK_SIZE 开始)       │
│  - l1ATensor[0/1]: P Ping-Pong (M×kDyn 个 ElementA)         │
│  - l1BTensor:      V 单缓冲 (stackSeqTile×embed 个 ElementB)│
└──────────────┬───────────────────────────────────────────────┘
               │ MTE1 (L1↔L0 DMA)
               ▼
┌──────────────────────────────────────────────────────────────┐
│ L0 (Cube 单元输入/输出缓冲)                                  │
│  - l0ATensor[0/1]: L0A (A 矩阵 Cube 格式, Ping-Pong)        │
│  - l0BTensor[0/1]: L0B (B 矩阵 Cube 格式, Ping-Pong)        │
│  - l0CTensor[0/1]: L0C (C 累加器/输出, Ping-Pong)           │
└──────────────┬───────────────────────────────────────────────┘
               │ Cube (MMAD 矩阵乘单元)
               ▼
         L0C 累加 → FIX 管道 → GM (gC)
```

### L1 内存布局图

```
L1 buffer (从 l1BufAddrStart 开始, 紧接 QK 区域之后)
┌─────────────────────────────┐ ← 0 (相对PV起始)
│  L1A[0] (P slot 0)          │
│  M × kDyn × sizeof(ElementA)│  例如 128×256×2B = 64KB
├─────────────────────────────┤ ← M*kDyn*sizeof(A)
│  L1A[1] (P slot 1)          │
│  M × kDyn × sizeof(ElementA)│  例如 128×256×2B = 64KB
├─────────────────────────────┤ ← M*kDyn*sizeof(A)*STAGES
│  L1B (V 单缓冲)              │
│  stackSeqTile × embed × ... │  例如 512×128×2B = 128KB
└─────────────────────────────┘
```

---

## 5. 类结构概览

```
BlockMmad<MmadAtlasA2FAIPVT<PAGED,ENABLE_UNIT>, L1<128,128,256>, L0<128,128,128>, ...>
│
├── 类型别名
│   ├── ElementA/B/C, LayoutA/B/C
│   ├── CopyGmToL1A/B, CopyL1ToL0A/B, CopyL0CToGm (各层 DataCopy)
│   └── LayoutAInL1/L0, LayoutBInL1/L0, LayoutCInL0
│
├── 常量
│   ├── STAGES = 2 (Ping-Pong 双缓冲)
│   ├── BLOCK_SIZE = 16 (MMAD 行对齐)
│   └── COORD_DIM0/1/2 = 0/1/2 (M=rowNum, N=embed, K=stackSeqTile)
│
├── 构造函数: 分配 L1/L0 缓冲
├── resetBlockStart(): 重置分页偏移
├──
├── 分页辅助函数
│   ├── getBlockShape(): 设置当前块 K 维长度
│   ├── getKVOffset() ×2: 非分页/分页模式计算 V 的 GM 偏移
│   ├── setBlockParam(): 计算页块参数（首页/末页残块）
│   └── updateBlockOffset(): 更新块内偏移
│
├── operator(): 主入口，四阶段执行
│   ├── 阶段1: V 预加载 GM→L1 (含分页分支)
│   ├── 阶段2: CrossCoreWaitFlag(softmaxFlag) 等待 Softmax
│   ├── 阶段3: 三层嵌套 GEMM 计算
│   │   ├── nL1 (embed, 步长128)
│   │   │   └── mL1 (Q行, 步长128)
│   │   │       ├── Wait FIX_M (等待C写回)
│   │   │       └── kL1 (seq, 步长l1KDynamic)
│   │   │           ├── Wait MTE1_MTE2 → GM→L1A(P) → Set MTE2_MTE1
│   │   │           └── kL0 (L0子块, 步长128)
│   │   │               ├── L1A→L0A, L1B→L0B
│   │   │               ├── Set/Wait MTE1_M(EVENT_ID0)
│   │   │               ├── tileMmad() [initMmad]
│   │   │               └── Set M_MTE1 (释放L0A/B)
│   │   └── Wait M_FIX → L0C→GM → Set FIX_M
│   └── 阶段4: Set MTE1_MTE2(EVENT_ID4) 排空信号
│
└── 成员变量
    ├── l1ATensor[2], l1BTensor, l0ATensor[2], l0BTensor[2], l0CTensor[2]
    ├── tileMmad, copyGmToL1A/B, copyL1ToL0A/B, copyL0CToGm
    ├── l1PPingPongFlag, l0CPingPongFlag, l0ABPingPongFlag
    └── l1M/N/KDynamic, blockStartOffset, maxKVStackLen
```

---

## 6. 四阶段执行流程详解

### 阶段 1：V 预加载 GM→L1

在 PV 计算开始前，先将整个 V 矩阵（当前 KV stack 的 Value）一次性搬运到 L1B。

**非分页模式** (`PAGED_CACHE_FLAG_=false`)：
- 单次 `copyGmToL1B(l1BTensor, gB[gBOffset], ...)` 将连续 V 数据搬运到 L1B。

**分页模式** (`PAGED_CACHE_FLAG_=true`)：
- V 被分成多个物理块（每页 blockSize=128 个 token），通过 `blockTable` 间接寻址；
- 逐页调用 `copyGmToL1B` 将各页 V 数据**拼接**到 L1B 的连续区域；
- 处理首页/末页残块（不完整页）。

```
分页 V 加载示意（blockSize=128, stackSeqTile=300）:

GM 中物理存储（非连续）:
  ┌──────┐  ┌──────┐  ┌──────┐
  │页[5] │  │页[12]│  │页[3] │   ← blockTable=[5,12,3,...]
  │128tok│  │128tok│  │44tok │
  └──────┘  └──────┘  └──────┘
     │          │          │
     └──────────┼──────────┘
                │ copyGmToL1B (3次调用)
                ▼
L1B (连续):
  ┌──────────────────────────────┐
  │ 页5  │ 页12  │ 页3残(44tok) │  共300个token连续排列
  │128tok│ 128tok│   + padding   │
  └──────────────────────────────┘
```

V 加载完成后，SetFlag/WaitFlag EVENT_ID0 确保 MTE2 管道完成。

### 阶段 2：跨核同步（等待 Softmax）

```c++
Arch::CrossCoreWaitFlag(softmaxFlag);  // softmaxFlag = SOFTMAX_READY_ID = 2
```

Cube 核阻塞，直到 Vector 核完成 online softmax 计算并通过 `Arch::CrossCoreSetFlag(softmaxReady)` 发出信号。此时 P 矩阵（softmax 输出）已写入 GM 的 workspace 区域，可以安全加载。

**信号链时序**：
```
Vector核:  online_softmax 计算 P
              │
              ├─── CrossCoreSetFlag(softmaxReady=2) ──┐
              │                                        │
Cube核:   ... V预加载 ... WaitFlag(EVENT_ID0)          │
              │                                        │
              └──── CrossCoreWaitFlag(softmaxFlag) ◀───┘  ← 阻塞到这里
                            │
                            ▼
                      P×V 计算开始
```

关键优化：V 预加载（阶段1）在等待 softmaxFlag **之前**就开始了，V 的 GM→L1 DMA 可以和 Vector 核的 softmax 计算**并行执行**，隐藏 DMA 延迟。

### 阶段 3：三层嵌套 GEMM 计算

#### 循环结构

```
nL1 (embed维, 步长128)          ← 最外层：输出列分块
  mL1 (Q行维, 步长128)          ← 中层：输出行分块
    Wait FIX_M                  ← 等待上一 mL1 块的 C 写回完成
    kL1 (seq维, 步长l1KDynamic) ← 内层：K维分块（P 的 Ping-Pong 粒度）
      Wait MTE1_MTE2            ← 等待 L1A slot 空闲
      GM→L1A: copyGmToL1A(P)   ← 搬运 P 子块 [mL1Actual × kL1Actual]
      Set MTE2_MTE1             ← 通知 P 已到 L1
      kL0 (L0子块, 步长128)     ← 最内层：Cube 计算粒度
        Wait M_MTE1(flag+0)     ← 等待 L0A slot
        Wait M_MTE1(flag+2)     ← 等待 L0B slot
        [kL0Idx==0] Wait MTE2_MTE1  ← 首子块额外等 P 完整到 L1
        L1A→L0A: copyL1ToL0A   ← P 子块 → L0A
        L1B→L0B: copyL1ToL0B   ← V 子块 → L0B
        [kL0Idx==last] Set MTE1_MTE2 ← 释放 L1A slot
        Set/Wait MTE1_M(EVENT_ID0)   ← 等 L0A/B 都就绪
        tileMmad(initMmad)      ← Cube 矩阵乘（首个k块初始化累加器）
        Set M_MTE1(flag+0)      ← 释放 L0A
        Set M_MTE1(flag+2)      ← 释放 L0B
        flag ^= 1               ← Ping-Pong 切换
    Wait M_FIX                  ← 等 Cube 完成
    L0C→GM: copyL0CToGm(O_tmp) ← 写回输出
    Set FIX_M                   ← 释放 L0C
    flag ^= 1
```

#### 事件同步链（单个 kL0 迭代内）

```
时间轴 ──────────────────────────────────────────────▶

MTE2(GM→L1): [====P GM→L1A====]
MTE1(L1→L0):    Wait   [==P L1→L0A==]        [==V L1→L0B==]
M(Cube):               Wait                       [===MMAD===]
FIX(C→GM):                                                Wait [==C L0→GM==]

事件:          │SetMTE2 │      │SetMTE1 WaitMTE1  │SetM WaitMTE1  │SetM     │SetM_FIX
               │_MTE1   │      │_M(0)     _M(2)  │_M(0)  _M(0)  │_MTE1(0) │
               │        │Wait  │                 │  WaitEVENT_ID0│_MTE1(2) │WaitM_FIX
               │        │MTE2  │Wait Wait        │               │flag^=1  │
               │        │_MTE1 │M_MTE1 M_MTE1    │               │         │
               │        │(flag)(flag) (flag+2)  │ MMAD          │         │
```

#### Ping-Pong 机制

- **l1PPingPongFlag** (0/1)：L1A 的 P slot 切换。每个 kL1 块 DMA 一个 P 子块到 L1A[flag]，同时 kL0 循环消耗上一个 P 子块（L1→L0），形成 DMA/计算重叠。
- **l0ABPingPongFlag** (0/1)：L0A/L0B slot 切换。flag=0 时使用 L0A[0]/L0B[0]，flag=1 时使用 L0A[1]/L0B[1]。L0A 用事件号 flag+0（0/1），L0B 用事件号 flag+2（2/3），避免冲突。
- **l0CPingPongFlag** (0/1)：L0C 的 C slot 切换。每个 mL1 块结束后写回并切换。

### 阶段 4：管道排空信号

```c++
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
```

标记 PV 阶段的 MTE1/MTE2 管道已完全排空，供下一次 PV 调用（下一个 KV stack）开始时的 WaitFlag(EVENT_ID4) 通过。

---

## 7. 分页 KV Cache 逻辑详解

分页模式下，V 矩阵不是连续存储的，而是分成固定大小的物理块（blockSize=128 tokens/块），通过 `blockTable`（形状 [batch×maxBlocks]）记录逻辑块→物理块的映射。

### 核心状态变量

- **blockStartOffset**：当前在物理块内的偏移位置。跨多个 PV 调用时保持，用于跟踪分页加载的连续性。
- **maxKVStackLen**：单个 KV stack 最大序列长度，用于计算逻辑块号 `nowNIdx = nIdx * maxKVStackLen / blockSize + curBlockIdx`。

### 分页加载步骤（setBlockParam 逻辑）

假设上次 PV 调用结束时 `blockStartOffset=0`（新的 stack 刚开始）：

1. `blockStart = blockSize - blockStartOffset = 128`（首页完整可用 128 tokens）
2. 若 stackSeqTile=300：
   - 300 >= 128，进入跨页分支
   - blockEnd = (300-128)%128 = 172%128 = 44（末页取 44 tokens）
   - curBlockTotalNum = ⌈(300-128)/128⌉ + 1 = ⌈172/128⌉ + 1 = 2+1 = 3 个物理块
3. 逐块加载：
   - curBlockIdx=0: nowLen=128-0=128, nowNIdx=..., 加载首页128 tokens → L1B[0:128]
   - curBlockIdx=1: nowLen=128-0=128 (非末页), 加载第二页128 tokens → L1B[128:256]
   - curBlockIdx=2: nowLen=44-0=44 (末页), 加载第三页44 tokens → L1B[256:300]

---

## 8. 例子 1：Prefill + Causal Mask + 非分页 KV

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| seqlen_q | 512 | Prefill 阶段序列长度 |
| seqlen_kv | 512 | KV 序列长度 |
| d_head (embed) | 128 | 注意力头维度 |
| qNBlockSize | 128 | Q 分块大小 |
| stackSeqTile | 256 | 当前 KV stack tile 长度 |
| PAGED_CACHE_FLAG | false | 连续 KV Cache |
| 数据类型 | fp16 | 半精度 |
| L1TileShape | `<128,128,256>` | M=128, N=128, K=256 |
| L0TileShape | `<128,128,128>` | Cube 基本计算块 |
| l1KDynamic | 256 | L1 K维步长 |

### 循环次数计算

- `nL1Loop = CeilDiv(128, 128) = 1`（embedding 不拆分）
- `mL1Loop = CeilDiv(128, 128) = 1`（一个Q块 128行）
- `kL1Loop = CeilDiv(256, 256) = 1`（一个K块 256列）
- `kL0Loop = CeilDiv(256, 128) = 2`（两个 L0 子块）

### 执行追踪

```
V预加载 (非分页):
  copyGmToL1B(l1BTensor, gB[nIdx*maxKVStackLen*strideV], ...)
  → L1B 获得 V[0:256, 0:128]，共256×128个fp16

CrossCoreWaitFlag(softmaxFlag): 等待Vector核softmax完成P

nL1Idx=0, nL1Actual=128:
  mL1Idx=0, mL1Actual=128:
    Wait FIX_M(0) → 通过（初始状态）
    kL1Idx=0, kL1Actual=256:
      Wait MTE1_MTE2(0) → 通过
      copyGmToL1A(l1A[0], gP[0:128, 0:256]) → GM→L1A
      Set MTE2_MTE1(0)

      kL0Idx=0, kL0Actual=128:
        Wait M_MTE1(0) → 通过
        Wait M_MTE1(2) → 通过
        [kL0Idx==0] Wait MTE2_MTE1(0) → 等待P的GM→L1A完成
        copyL1ToL0A(l0A[0], l1A[0][0:128,0:128]) → P_00
        copyL1ToL0B(l0B[0], l1B[0:128,0:128]) → V_00
        Set/Wait MTE1_M(EVENT_ID0)
        initMmad = true (kL1=0,kL0=0)
        mL0Align = (128+15)/16*16 = 128
        tileMmad(l0C[0], l0A[0], l0B[0], 128, 128, 128, init=true)
          → L0C[0] = P_00 × V_00  （初始化累加器）
        Set M_MTE1(0), Set M_MTE1(2), l0ABFlag=1

      kL0Idx=1, kL0Actual=128:
        Wait M_MTE1(1) → 通过
        Wait M_MTE1(3) → 通过
        [kL0Idx!=0] 不等待L1A
        copyL1ToL0A(l0A[1], l1A[0][0:128,128:256]) → P_01
        copyL1ToL0B(l0B[1], l1B[128:256,0:128]) → V_01
        [kL0Idx==last=1] Set MTE1_MTE2(0) → 释放L1A slot0
        Set/Wait MTE1_M(EVENT_ID0)
        initMmad = false
        tileMmad(l0C[0], l0A[1], l0B[1], 128, 128, 128, init=false)
          → L0C[0] += P_01 × V_01  （累加到前一结果）
        Set M_MTE1(1), Set M_MTE1(3), l0ABFlag=0
      l1PPingPongFlag=1

    Set/Wait M_FIX(EVENT_ID0) → 等待Cube完成
    copyL0CToGm(gOTmp[0:128,0:128], l0C[0]) → L0C→GM 写回 OTmp
    Set FIX_M(0), l0CPingPongFlag=1

Set MTE1_MTE2(EVENT_ID4) → 排空完成
```

### 数据流图

```
          K=256 (stackSeqTile)
          ┌─────────────────────┐
          │         V           │ N=128
M=128  P  │  P_00(128×128)      │
(q rows)  │  P_01(128×128)      │
          └─────────────────────┘
                    ×
          ┌─────────────────────┐
          │    V_00(128×128)    │
          │    V_01(128×128)    │
          └─────────────────────┘
                    ↓ MMAD
          ┌─────────────────────┐
          │     OTmp(128×128)   │
          │ = P00×V00 + P01×V01 │
          └─────────────────────┘

kL0=0: tileMmad(init=true)  → L0C = P_00 × V_00
kL0=1: tileMmad(init=false) → L0C += P_01 × V_01
```

---

## 9. 例子 2：Decode + GQA + 分页 KV Cache + bf16

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| seqlen_q | 1 | Decode 阶段（单 token 生成） |
| seqlen_kv | 2048 | KV Cache 总长度 |
| d_head (embed) | 128 | 注意力头维度 |
| groupQueryRatio | 8 (GQA) | Q头数:KV头数=8:1 |
| qNBlockSize | 1 | Decode 模式，sub-core 沿 seq 拆分 |
| stackSeqTile | 512 | 每个 PV 调用处理 512 个 KV token |
| PAGED_CACHE_FLAG | true | 分页 KV Cache |
| blockSize | 128 | 每页 128 tokens |
| maxKVStackLen | 2048 | 最大 KV 长度 |
| 数据类型 | bf16 | 推理常用精度 |
| l1KDynamic | 256 | L1 K维步长 |

### 场景描述

Decode 阶段生成第 2048 个 token，当前处理第 0 个 KV stack (nIdx=0)，对应 KV 位置 [0,512)。P 矩阵是一个行向量（1 行 × 512 列，softmax 后权重），V 矩阵是 [512, 128]。

### V 分页加载追踪

blockTable 前 4 个逻辑块（页）映射到物理块 [5, 12, 3, 8]（假设）。

- `blockStartOffset = 0`（stack 起始）
- `blockStart = 128 - 0 = 128`
- `stackSeqTile=512 >= 128, blockSize=128`：
  - `blockEnd = (512-128)%128 = 384%128 = 0`？→ 实际 blockEnd=128（因为能整除）
  - `curBlockTotalNum = ⌈384/128⌉ + 1 = 3 + 1 = 4` 个物理块

```
curBlockIdx=0: nowLen=128, nowNIdx=0*16+0=0, blockTable[0]=5
              kOffset=5*128*strideV+0, copy到L1B[0:128]
              blockStartOffset=0+128=128 → 正好blockSize → 归零, curBlockIdx=1

curBlockIdx=1: nowLen=128, nowNIdx=1, blockTable[1]=12
              kOffset=12*128*strideV+0, copy到L1B[128:256]
              blockStartOffset=0, curBlockIdx=2

curBlockIdx=2: nowLen=128, nowNIdx=2, blockTable[2]=3
              kOffset=3*128*strideV+0, copy到L1B[256:384]
              blockStartOffset=0, curBlockIdx=3

curBlockIdx=3: nowLen=128 (blockEnd-blockStartOffset=128-0=128),
              nowNIdx=3, blockTable[3]=8
              kOffset=8*128*strideV+0, copy到L1B[384:512]
              blockStartOffset=0, curBlockIdx=4 → 退出循环
```

V 在 L1B 中连续排列为 [512×128] bf16，来自 4 个非连续物理页。

### GEMM 计算追踪

- `mL1Loop = CeilDiv(1, 128) = 1`，mL1Actual=1
- `nL1Loop = CeilDiv(128, 128) = 1`，nL1Actual=128
- `kL1Loop = CeilDiv(512, 256) = 2`
- `kL0Loop = CeilDiv(256, 128) = 2`（每个kL1块内）

MMAD 调用：1×1×2×2 = 4 次

```
nL1Idx=0, mL1Idx=0, mL1Actual=1:
  mL0Align = (1+15)/16*16 = 16  ← 注意对齐到16行

  kL1Idx=0, kL1Actual=256:
    Wait MTE1_MTE2(0)
    copyGmToL1A(l1A[0], P[0,0:256]) → 1×256 bf16
    kL0Idx=0: copyL1→L0, tileMmad(l0C[0], ..., init=true, mL0Align=16, kL0Actual=128)
              → L0C[0] = P[0:128] × V[0:128,:] (16×128×128, 有效行仅1行)
    kL0Idx=1: copyL1→L0, tileMmad(l0C[0], ..., init=false, kL0Actual=128)
              → L0C[0] += P[128:256] × V[128:256,:]
    l1PPingPongFlag=1

  kL1Idx=1, kL1Actual=256:
    Wait MTE1_MTE2(1)
    copyGmToL1A(l1A[1], P[0,256:512]) → 1×256 bf16
    kL0Idx=0: copyL1→L0, tileMmad(l0C[0], ..., init=false, kL0Actual=128)
              → L0C[0] += P[256:384] × V[256:384,:]
    kL0Idx=1: copyL1→L0, tileMmad(l0C[0], ..., init=false, kL0Actual=128)
              → L0C[0] += P[384:512] × V[384:512,:]
    l1PPingPongFlag=0

  Wait M_FIX, copyL0CToGm(gOTmp[0,0:128]) → 1×128 输出向量
  Set FIX_M(0)
```

### Paged V 加载图示

```
GM 物理存储（blockSize=128 tokens/页）:
  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐
  │页0 │  │页3 │  │页5 │  │页8 │  │页12│ ... (物理页，任意顺序)
  └────┘  └────┘  └────┘  └────┘  └────┘

blockTable (逻辑→物理映射):
  [0]→5, [1]→12, [2]→3, [3]→8, [4]→...

加载过程（4次DataCopy，拼成L1B连续区域）:
  ┌────┐    ┌────┐    ┌────┐    ┌────┐
  │页5 │ ─┐ │页12│ ─┐ │页3 │ ─┐ │页8 │ ─┐
  └────┘  │ └────┘  │ └────┘  │ └────┘  │
          ▼         ▼         ▼         ▼
  L1B: ┌──────┬──────┬──────┬──────┐
       │ 0-127│128-255│256-383│384-511│
       └──────┴──────┴──────┴──────┘
       tok0~127 tok128~255 tok256~383 tok384~511
```

---

## 10. 设计亮点

1. **镜像 Ping-Pong 策略**：与 QK 对偶——V 预加载常驻 L1（因 V 在 stack 内固定），P 动态 Ping-Pong（因 P 由 Vector 核逐块产出），最大化数据复用。
2. **V 预加载与 Softmax 并行**：V 的 GM→L1 DMA 在 CrossCoreWaitFlag 之前启动，与 Vector 核的 softmax 计算并行，隐藏 DMA 延迟。
3. **分页 KV Cache 支持**：通过 `if constexpr (PAGED_CACHE_FLAG_)` 编译期分支，零开销支持连续/分页两种布局；分页模式下逐页 DataCopy 并拼接为 L1 连续区域，后续 GEMM 计算无需感知分页。
4. **三层 Ping-Pong 精细流水线**：L1A (P)、L0A/L0B (P/V 子块)、L0C (C 输出) 三级独立 Ping-Pong，各用独立事件号，最大化 DMA/计算重叠。
5. **跨核同步点最小化**：仅一个 CrossCoreWaitFlag（等待 softmaxReady），其他全为核内 HardEvent 同步，延迟可控。
6. **EVENT_ID4 排空协议**：使用独立事件号 PV 入口/出口的管道排空，支持多 KV stack 连续调用时的安全复用。
7. **尾块处理**：各维循环中通过 `idx < Loop-1 ? TileSize : remainder` 正确处理非对齐尾块；Cube 行对齐通过 `mL0Align` 显式填充到 BLOCK_SIZE=16。
8. **initMmad 精确控制**：仅在每个 (mL1, nL1) 的首个 k 子块初始化累加器，后续 k 块全部累加，确保跨 k 块归约正确。

---

## 11. 注意事项

- `ENABLE_UNIT_FLAG_` 模板参数已声明但当前版本未使用（预留）。
- `EMBED_SPLIT_SIZE`, `UNIT_BLOCK_STACK_NUM`, `KV_BASE_BLOCK`, `KV_SPLIT_SIZE`, `LOAB_BLOCK` 为预留常量，当前代码未引用。
- `l1MDynamic` 成员赋值为 0 但未使用（预留）。
- L1A 分配使用 `L1TileShape::M * kDyn`（动态 kDyn），而非 L1A_SIZE 常量（用 L1TileShape::K=256），实际缓冲大小取决于实例化时传入的 kPVDynNum。
- `blockStartOffset` 在多个 KV stack 连续调用时保持状态，通过 `resetBlockStart()` 在新序列开始时清零。
- 分页加载中 `curBlockSize` 的计算在 curBlockIdx>0 时为 `(curBlockIdx-1)*blockSize + blockStart`，将首页残块长度正确计入 L1B 偏移。

---

## 12. 总结

`pv_matmul.hpp` 以约 300 行代码实现了 FlashAttention PV 阶段的高效 Cube GEMM，核心设计围绕以下三点展开：

1. **数据复用驱动的缓冲策略**：V 常驻 L1、P Ping-Pong 动态加载，与 QK 形成镜像对称；
2. **多级 Ping-Pong 流水线**：L1/L0 三级独立双缓冲，通过 HardEvent 精细同步，最大化 DMA-Cube 并行；
3. **分页透明化**：编译期分支处理分页/非分页，分页模式在 V 加载阶段完成物理→逻辑地址转换，后续 GEMM 计算路径完全统一。

PV GEMM 的输出 OTmp 经 `rescale_o.hpp` 跨 stack 累加并归一化后，得到最终的注意力输出 O，完成整个 FlashAttention 前向计算。
