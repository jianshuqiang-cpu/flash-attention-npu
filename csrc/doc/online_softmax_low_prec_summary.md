# online_softmax_low_prec.hpp 详解

## 1. 文件定位

`online_softmax_low_prec.hpp` 是 FlashAttention NPU 前向推理 pipeline 中 **Vector 核半精度(FP16) Online Softmax epilogue** 的 CATLASS BlockEpilogue 偏特化实现。

- **物理路径**：`csrc/flash_attn_npu/online_softmax_low_prec.hpp`
- **命名空间**：`Catlass::Epilogue::Block`
- **偏特化签名**：`BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>, ...>`
- **对应高精度版本**：[online_softmax.hpp](file:///d:/jianshuqiang/code/flash-attention-npu/csrc/flash_attn_npu/online_softmax.hpp)（`SM_DTYPE_=float`）
- **当前启用状态**：**预留未启用** —— 所有 flash_api.cpp 中的 kernel 实例化均使用 `IntermCalcPrec=float`（即高版本），本 half 版本保留作为低精度优化路径。

## 2. 在前向 Pipeline 中的位置

```
┌──────────────────────────────────────────────────────────────────────┐
│                        单个 AI Core (A2)                            │
│                                                                      │
│  ┌─────────────────┐         ┌──────────────────────────────┐        │
│  │   Cube Core     │         │      Vector Core (×2)        │        │
│  │                 │         │                              │        │
│  │  BlockMmadQK    │ qkReady │  BlockEpilogueOnlineSoftmax  │ softmax│
│  │  (Q × K^T = S) ─┼────────►│  (本文件)                    ├───────►│──► PV matmul
│  │                 │         │                              │ Ready  │
│  │                 │         │  ① Scale S by 1/√d           │        │
│  │                 │         │  ② Apply causal mask(-6e4)   │        │
│  │                 │         │  ③ Online softmax(m/l/P)     │        │
│  │                 │         │  ④ Write P to GM (workspace) │        │
│  └─────────────────┘         └──────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

- **上游生产者**：Cube 核的 `BlockMmadQK` 完成 Q*K^T 矩阵乘，将 S 写入 workspace（GM 中的 gS 区域），然后 `CrossCoreSetFlag(qkReady)`。
- **本 epilogue**：Vector 核等待 qkReady，从 GM 读取 S，在 UB 中完成 scale、mask、online softmax，将 P_tile 写回 GM（gP 区域），并在 UB 中维护跨 tile 的 m/l 累积量。
- **下游消费者**：Cube 核的 `BlockMmadPV` 读取 P 计算 P*V=OTmp；Vector 侧的 `rescale_o.hpp` epilogue 利用跨 tile 的 m/l 对 O 做重缩放累加。

## 3. Online Softmax 数学原理

普通 softmax 需要读取完整的 S 矩阵才能做归一化，这要求 O(N²) 的 HBM 带宽。FlashAttention 把 K/V 沿序列维切成多个 tile（每个 tile 宽 `MAX_KV_STACK_LEN=512`），逐 tile 流式计算，使用 **online softmax** 维护数值稳定的累积统计量：

### 3.1 三个核心量

设当前处理第 t 个 KV tile，S_t 形状为 (M, Bb)（M=qSBlockSize=128 行，Bb=当前 tile 列宽）：

| 符号 | 含义 | 存储位置 |
|------|------|----------|
| **m_t** | 行最大值（max）：`m_t = max(m_{t-1}, rowmax(S_t))` | UB `gmUbTensor`（跨 tile 保留） |
| **l_t** | 归一化分母（sum of exp）：`l_t = exp(m_{t-1}-m_t)*l_{t-1} + rowsum(exp(S_t - m_t))` | UB `glUbTensor`（跨 tile 保留） |
| **P_t** | 当前 tile 的概率矩阵：`P_t = exp(S_t - m_t)` | GM `gP`（写出给 PV matmul） |

### 3.2 六步计算流（SubCoreCompute 内部）

```
① CalcLocalRowMax:      m_local = rowmax(S_t)               → lm
② UpdateGlobalRowMax:   m_new  = max(m_local, m_old)        → hm
                         dm     = exp(m_old - m_new)         → dm (rescale系数)
                         m_old ← m_new                       → gm←hm
③ CalcExp:              P_t    = exp(S_t - m_new)           → compute区
④ MoveP + CopyPUbToGm:  P_t → lp → GM(gP)  (给Cube PV用)
⑤ CalcLocalRowSum:      l_local = rowsum(P_t)               → ll
⑥ UpdateGlobalRowSum:   l_new = dm * l_old + l_local        → gl
```

### 3.3 为什么不需要在本 epilogue 中归一化 P？

P_t = exp(S_t - m_new) 是**未归一化**的概率（缺少除以 l_new）。这是 FlashAttention 的精妙之处：

- 后续 PV matmul 计算 O_t = P_t * V_t
- `rescale_o.hpp` epilogue 中：`O_new = exp(m_old-m_new)*O_old + O_t`，最后除以 l_final 完成归一化
- 延迟归一化避免了每个 tile 内的除法，节省算力且与 O 的 rescale 融合

## 4. low_prec (half) vs high_prec (float) 差异

| 维度 | low_prec (half，本文件) | high_prec (float) |
|------|------------------------|-------------------|
| **中间精度** | FP16 (half) | FP32 (float) |
| **S 区 UB 容量** | 16384 half = 32KB | 8192 float = 32KB |
| **S 计算缓冲** | 独立 `computeUbTensor`，ls 仅作着陆 | 直接在 `lsUbTensor[sUbOffset]` 原地计算 |
| **mask16 复用** | 复用 LS 区 0 偏移（S 已搬到 compute） | 独立 offset=11×16KB |
| **mask UpCast** | int8→half（一步） | int8→half→float（两步） |
| **mask 填值** | (half)-6e4 ≈ -60000 | (float)-3e38 |
| **P 输出** | 直接 DataCopy（half→half） | 需要 Cast<float→half/bf16> |
| **行规约** | Add + WholeReduceSum（两级） | BlockReduceSum 三级级联 |
| **256 列特化** | 无 | 有 Rowsum/Rowmax SPECTILE256 |
| **softcap 支持** | 不支持（参数被注释） | 支持 tanh softcap |
| **preLoad 预取** | 无（串行） | preLoad=1（DMA 与计算重叠） |
| **isLastNoMaskStackTile** | 接收但未使用 | 用于流水排空同步 |
| **当前使用情况** | 预留（未实例化） | 所有 kernel 均使用 |

## 5. UB 内存布局

Atlas A2 每个 Vector 核有 ~192KB Unified Buffer (UB)，low_prec 版本的分区如下：

```
 字节偏移  大小   张量                 用途
────────────────────────────────────────────────────────────
 0KB       32KB   lsUbTensor           S 原始输入 (GM→UB 着陆区, half)
                   ↳ maskUbTensor16     mask UpCast 后复用此区 (half)
 32KB      32KB   computeUbTensor      S 计算区 (scale/mask/exp, half, 双缓冲pingpong)
 64KB      16KB   lpUbTensor           P 输出缓冲 (half/bf16, 待写GM)
 80KB-160KB  (gap / 其他Reserved)
 160KB     8KB    tvUbTensor           Brcb 广播 / 规约临时 (half)
 168KB     1KB    lmUbTensor           局部行 max (m_local)
 169KB     1KB    hmUbTensor           新全局行 max (m_new)
 170KB     1KB    gmUbTensor           旧全局行 max (m_old)
 171KB     1KB    llUbTensor           局部行 sum (l_local)
 172KB     1KB    glUbTensor           全局行 sum (l_new)
 173KB     1KB    dmUbTensor           delta-m = exp(m_old-m_new)
 174KB-176KB (padding)
 176KB     16KB   maskUbTensor         int8 mask (GM→UB, DataCopyPad 目标)
 192KB
```

总占用约 32+32+16+8+6+16 ≈ 110KB（不含 gap），为 Resource 管理的其他预留区留出空间。

## 6. 类结构与成员函数

```
BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, half>, ...>
│
├── 常量定义
│   ├── BLOCK_SIZE=16 (half元素/block), HALF_VECTOR_SIZE=128, BLOCK_SIZE_IN_BYTE=32
│   ├── MAX_UB_S_ELEM_NUM=16384 (单S缓冲槽half元素数 = 32KB)
│   ├── MAX_ROW_NUM_SUB_CORE=256 (子核最大行数)
│   └── UB_UINT8_BLOCK_SIZE=16384, UB_UINT8_VECTOR_SIZE=1024
│
├── 构造函数
│   └── 从 resource.ubBuf 按固定字节偏移分配所有 LocalTensor
│
├── 工具函数
│   ├── SetVecMask(len)                  —— 设置 Vector 谓词掩码（处理尾块）
│   ├── SetBlockReduceMask(len)          —— 块归约专用掩码（当前版本未调用）
│   └── UpCastMask<Dst,Src>(...)         —— int8→half 类型提升
│
├── 数据搬运
│   ├── CopySGmToUb(...)                 —— S: GM→UB(ls区)
│   ├── CopyMaskGmToUb(...)              —— mask: GM→UB(mask区)，分prologue/integral/epilogue三段
│   ├── MoveP(...)                       —— P: compute区→lp区
│   └── CopyPUbToGm(...)                 —— P: lp区→GM(gP)
│
├── S 预处理
│   ├── ScaleS(...)                      —— S = scaleValue * S (=1/√d)
│   └── ApplyMask(...)                   —— S += -6e4 * mask（mask位置→-∞）
│
├── Online softmax 核心计算
│   ├── CalcLocalRowMax(...)             —— m_local = rowmax(S_t)
│   │   └── RowmaxTAILTILE(...)          —— 通用行max（含尾块掩码）
│   ├── UpdateGlobalRowMax(...)          —— m_new=max(m_local,m_old), dm=exp(m_old-m_new)
│   ├── CalcExp(...)                     —— P_t = exp(S - m_new)，含Brcb广播
│   ├── CalcLocalRowSum(...)             —— l_local = rowsum(P_t)
│   │   ├── RowsumSPECTILE512(...)       —— 512列快速路径(4块Add→WholeReduceSum)
│   │   └── RowsumTAILTILE(...)          —— 通用路径(循环Add+WholeReduceSum)
│   └── UpdateGlobalRowSum(...)          —— l_new = dm*l_old + l_local
│
├── SubCoreCompute(...)                  —— 单个子核上的完整softmax调度
│
└── operator() (两个重载)
    ├── 重载1：无mask (5个tile参数)       —— NO_MASK 或 完全在对角线下
    └── 重载2：带causal mask (+gMask,qkReady,triUp/triDown等)
```

## 7. 两个 operator() 重载对比

| 项 | 重载1（无 mask） | 重载2（带 causal mask） |
|----|-----------------|------------------------|
| **触发场景** | NO_MASK 编译期路径；或 causal+完全在对角线下方 | causal+跨越对角线（triUp < kvSEndIdx-1） |
| **qkReady 等待** | 调用者外部 CrossCoreWaitFlag | 内部第一行 CrossCoreWaitFlag |
| **mask 参数** | 无 | gMask, layoutMask, triUp, triDown, kvSStartIdx, kvSEndIdx |
| **mask 相关步骤** | 无 | CopyMaskGmToUb → UpCastMask → ApplyMask |
| **mask 偏移计算** | 无 | 根据 triUp 是否在 stack 内计算 gmOffsetMaskRow/Column/maskColumn/addMaskUbOffset |
| **行循环额外逻辑** | CopyS→ScaleS→SubCoreCompute | + CopyMask/UpCast/ApplyMask；mask 用 EVENT_ID1/ID3 独立流水 |
| **isLastNoMaskStackTile** | 接收（但未使用） | 不接收 |

### 7.1 子核任务拆分（两个重载共用）

Atlas A2 每个 Vector 核有 2 个 sub-core（sub-core 0 和 sub-core 1），需要把所有行拆分给它们并行处理。行总数 `rowNum = qSBlockSize × qNBlockSize`。

```
情况A: qNBlockSize == 1 (decode 单头)
  → 按序列维对半拆：rowSplitSubBlock = qSBlockSize/2
  ┌─────────────── qSBlockSize=128行(1个head) ──────────────┐
  │ sub-core0: 行 0~63  │  sub-core1: 行 64~127              │
  └─────────────────────┴────────────────────────────────────┘
  maskOffsetThisSubBlock = rowOffsetThisSubBlock（mask行偏移不同）

情况B: qNBlockSize > 1 (prefill 多头并行)
  → 按 head 维拆：rowSplitSubBlock = qSBlockSize × qNSplitSubBlock
  ┌── qNSplitSubBlock 个head × 128行 ──┬── 剩余 head × 128行 ──┐
  │          sub-core0                 │      sub-core1          │
  └────────────────────────────────────┴─────────────────────────┘
  maskOffsetThisSubBlock = 0（mask 靠 prologue/integral/epilogue 处理多头）
```

### 7.2 行分批循环

UB 的 S 单槽容量为 16384 half = 32KB，因此一次最多处理 `maxRowNumPerLoop = 16384 / columnNumRound` 行，再向下对齐到 BLOCK_SIZE，且不超过 64 行：

- columnNum=512（完整 KV tile）：`maxRowNumPerLoop = 16384/512 = 32 行`
- columnNum=128（尾块）：`maxRowNumPerLoop = 16384/128 = 128 → min(128,64) = 64 行`

使用 `pingpongFlag = rowLoopIdx % 2` 实现双缓冲（ls/compute 的两槽交替）。

## 8. 核内/核间同步机制

### 8.1 跨核同步（Cube ↔ Vector）

| 事件 | 设置方 | 等待方 | 作用 |
|------|--------|--------|------|
| `qkReady` (ID=1) | Cube (CrossCoreSetFlag) | Vector (CrossCoreWaitFlag) | Q*K^T 完成，S 已写到 GM |
| `softmaxReady` (ID=2) | Vector (CrossCoreSetFlag) | Cube (CrossCoreWaitFlag) | softmax 完成，P 已写到 GM，PV 可以开始 |

### 8.2 核内流水（MTE2 ↔ V ↔ MTE3）

Vector 核内部通过 `SetFlag/WaitFlag<HardEvent>` 在不同 pipe 间建立软流水：

```
EVENT_ID0 (主流水 S搬运↔计算↔P写回)
  WaitFlag(V_MTE2) → CopyS(GM→UB) → SetFlag(MTE2_V) → WaitFlag(MTE2_V) → ScaleS
                                                          ...
  CalcExp → SetFlag(V_MTE2)  // 通知 CalcLocalRowMax 后 UpdateGlobalRowMax 可安全访问 lm?
  WaitFlag(MTE3_V) → MoveP → SetFlag(V_MTE3) → CalcLocalRowSum
  WaitFlag(V_MTE3) → CopyP(UB→GM) → SetFlag(MTE3_V) → UpdateGlobalRowSum

EVENT_ID1 (mask UpCast 完成信号)
EVENT_ID3 (mask GM→UB 完成 → 允许下一轮覆盖 mask 区)
EVENT_ID4 (LSE 专用，OUT_ONLY 模式首块等待 LSE 数据就绪)
```

### 8.3 SubCoreCompute 内部流水时序

```
时间 →
CalcLocalRowMax ─┐
                 ├─ V_MTE2↓EVENT_ID0 (通知UpdateGlobalRowMax开始)
UpdateGlobalRowMax┤
CalcExp ─────────┴────────────────────────────────────────────┐
                                                              ├─ MTE3_V↑EVENT_ID0 (等上一轮P写回完成)
WaitFlag(MTE3_V) ─────────────────────────────────────────────┘
MoveP(compute→lp) → V_MTE3↓EVENT_ID0 (通知CalcLocalRowSum后可写GM)
CalcLocalRowSum ─┐
                 ├─ V_MTE3↑EVENT_ID0 (等MoveP完成)
WaitFlag(V_MTE3)─┘
CopyPUbToGm(lp→GM) → MTE3_V↓EVENT_ID0 (通知下轮WaitFlag可MoveP)
UpdateGlobalRowSum (与下一轮CalcLocalRowMax并行?)
```

## 9. 例子1：Prefill + Causal Mask + 完整 512 列块

### 9.1 场景参数
- batch=1, seqlen=1024, num_heads=32, head_dim=128（half精度）
- qSBlockSize=128, MAX_KV_STACK_LEN=512, scaleValue=1/√128≈0.0884
- qNBlockSize=1（prefill单头）, 所以每子核分 128/2=64 行
- 当前处理 Q block 的行 [512,640)（sub-core0）/ [640,768)（sub-core1），KV tile 列 [0,512)
- noSkipKvS=qBlock起始+qSBlockSize=512+128=640；triUp=noSkipKvS-qSBlockSize=512；triDown=640
- kvSStartIdx=0, kvSEndIdx=512 → triUp(512) >= kvSStartIdx(0) → 走**带mask重载**

### 9.2 mask 区域计算

```
triUpRoundDown = RoundDown(512, 16) = 512
gmOffsetMaskRow = 512 - 512 = 0
gmOffsetMaskColumn = 0
maskColumn = 512 - 512 = 0   (! maskColumn=0 → 实际无mask列)
addMaskUbOffset = 512 - 0 = 512
```

等等，maskColumn=0 意味着实际上这个 tile 全部在可见区域——这是因为 triUp=512 正好在 tile 边界。若当前 KV tile 为列 [512,1024)，kvSStartIdx=512, kvSEndIdx=1024：
```
triUp(512) >= kvSStartIdx(512)
triUpRoundDown = 512
maskColumn = 1024 - 512 = 512（整块mask）
addMaskUbOffset = 512 - 512 = 0
```
这种情况下走整块 mask 应用。实际跨越对角线的 tile 是列 [0,512) 对于 Q 行 [512,640)：
- 行512可见KV≤512，行513可见KV≤513，...，行639可见KV≤639
- Q行[512,640)对KV[0,512)：全部可见（因所有Q位置≥512，KV位置≤511）→ doTriUMask=true但实际maskColumn很小

让我取更典型的边界例子：Q行[384,512)，KV列[384,512)（对角线穿过此块）：
- noSkipKvS = 512（行511可见KV≤511）
- triUp = 512-128 = 384, triDown=512
- kvSStartIdx=384, kvSEndIdx=512
- triUpRoundDown=384
- maskColumn=512-384=128
- addMaskUbOffset=0
- 对于行 r∈[384,512)（子核0: 384~448, 子核1: 448~512），从列 (r-384) 开始mask（即列≥r+1-384的位置）

此时 maskColumn=128，ApplyMask 走非整块路径（maskColumnRound=128 == columnNumRound=128？若columnNum=128则整块mask）。

### 9.3 数据处理流（sub-core 0，列数=128 这种情况）

```
1. CopyS:   GM(gS @ 行384~448, 列384~512) → lsUbTensor (64×128 half = 16KB)
2. ScaleS:  ls → compute, S = 0.0884 * S
3. CopyMask: GM(mask @ 行384~448, 列384~512) → maskUbTensor(int8)
4. UpCast:  int8 → half (到 maskUbTensor16, 复用ls区)
5. Muls:    mask16 *= -6e4 → 每个mask元素变-60000或0
6. ApplyMask: compute[addMaskUbOffset...] += mask16 (需mask的列加-6e4)
7. CalcLocalRowMax: rowmax(compute) → lm (每行1个half)
8. UpdateGlobalRowMax:
     isFirstStackTile? 否（Q块[384,512)的第1个KV tile是列[0,128)）
     hm = max(lm, gm)
     dm = exp(gm - hm)
     gm ← hm
9. CalcExp: tv=Brcb(hm), compute = exp(compute - tv)
10. WaitFlag(MTE3_V → EVENT_ID0)
11. MoveP: compute → lp
12. SetFlag(V_MTE3)
13. CalcLocalRowSum: RowsumTAILTILE(column=128, 单WholeReduceSum) → ll
14. WaitFlag(V_MTE3)
15. CopyPUbToGm: lp → GM(gP @ 行384~448, 列384~512)
16. SetFlag(MTE3_V)
17. UpdateGlobalRowSum: gl = dm*gl + ll
```

## 10. 例子2：Decode + 无 mask + 尾块

### 10.1 场景参数
- Decode 阶段：batch=8, seqlen=1（每batch 1个query）, num_heads=32, head_dim=128
- GQA：qHeads=32, kvHeads=32→groupSize=1, qNBlockSize=128（decode大head批）
- qSBlockSize=1（单token），rowNum = qSBlockSize × qNBlockSize = 128
- 当前 KV tile 列 [0,128)，完全在对角线下方（query能看到所有KV）→ doTriUMask=false → 走**无mask重载**
- 假设 KV 总长度=200，所以第二个KV tile（列[128,200)）是尾块，columnNum=72

### 10.2 子核拆分
- qNBlockSize=128（>1），按 head 维拆分
- qNSplitSubBlock = 128/2 = 64 head/子核
- rowSplitSubBlock = 1×64 = 64 行/子核
- sub-core0: 64行（64个head），sub-core1: 128-64=64行

### 10.3 尾块列数=72 处理

columnNum=72, columnNumRound=RoundUp(72,16)=80, columnNumPad=128(stride)

```
CalcLocalRowMax（RowmaxTAILTILE, numElems=72<=128）:
  SetVecMask(72)                    // 前72个half位有效
  WholeReduceMax(ls, compute, 72)   // 仅对前72个元素取每行max
  SetVectorMask(-1,-1)              // 恢复全1掩码

CalcExp 尾块处理:
  for subIdx in [0, 72/128)=0次: (无完整块)
  尾块: SetVecMask(72%128=72)
       Sub(compute[0], compute[0], tv)  // 减行max，尾块掩码保护
       SetVectorMask(-1,-1)
  Exp: (rowNumCurLoop*80+127)/128 = (64*80+127)/128 = 40 repeats

CopyPUbToGm: columnNumPad-columnNumRound=128-80=48个half(96字节)的列间隔padding
```

### 10.4 maxRowNumPerLoop 计算
```
maxRowNumPerLoop = 16384 / 80 = 204.8 → RoundDown(204.8,16)=192 → min(192,64)=64行
```
每子核64行一批完成。

## 11. ASCII 图

### 11.1 Online Softmax 跨 tile 数据流
```
Tile0 (KV列[0,Bb))                    Tile1 (KV列[Bb,2Bb))
─────────────────────                ─────────────────────
S0 = Q*K0^T (scale)                   S1 = Q*K1^T (scale)
  │                                    │
  ▼                                    ▼
m0 = rowmax(S0)                       m1_local = rowmax(S1)
l0 = rowsum(exp(S0-m0))               m1 = max(m0, m1_local)
P0 = exp(S0-m0) ──► GM(gP0)          dm = exp(m0-m1)
  │                                   P1 = exp(S1-m1) ──► GM(gP1)
  │                                   l1 = dm*l0 + rowsum(P1)
  │ (rescale_o最终使用)                │ (继续累积)
  ▼                                   ▼
 gl=m0, gl=l0 ───────────────►        gm=m1, gl=l1 ──────► ...下一个tile
```

### 11.2 单个子核 SubCoreCompute 数据通路
```
                ┌────────── UB ──────────┐
                │                        │
GM(gS) ─MTE2─► lsUbTensor               │
                │                        │
                ▼                        │
           compute ◄── scaleValue        │
                │                        │
     (mask路径) ▼                        │
         compute += -6e4*mask            │
                │                        │
                ├──► lm (rowmax)         │
                │       │                │
                │       ▼                │
                │     hm=max(lm,gm) ─┐   │
                │     dm=exp(gm-hm) │   │
                │     gm←hm         │   │
                │       │           │   │
                │       ▼           │   │
                │     tv=Brcb(hm)   │   │
                │       │           │   │
                ▼       ▼           │   │
         compute=exp(S-tv)          │   │
                │                   │   │
                ├─►MTE3─►lp─►MTE3──┼───┼──► GM(gP)
                │                   │   │
                ├──► ll (rowsum)    │   │
                │       │           │   │
                │       ▼           │   │
                │     gl=dm*gl+ll   │   │
                │     (UpdateGlobalRowSum)
                └───────────────────┴───┘
```

### 11.3 Causal Mask 三角区域（addMaskUbOffset 示意）
```
KV列 →  kvSStartIdx=384         对角线    kvSEndIdx=512
        │                           │           │
Q行 ▼   ▼                           ▼           ▼
  384 ┌───────────────────────────────┬───────────┐
      │ 可见 (P=softmax)              │  mask(-∞) │ ← addMaskUbOffset=0, maskColumn=128
  416 ├───────────────────────────────┼───────────┤
      │ 可见     ...                  │    ...    │
  448 ├───────────────────────────────┼───────────┤ ← sub-core boundary
      │ 可见                          │  mask     │
  480 ├───────────────────────────────┼───────────┤
      │ 可见     对角线上三角▶        │  mask     │
  512 └───────────────────────────────┴───────────┘
      ◄── addMaskUbOffset=0 ────────►
      ◄──── maskColumn=128 ──────────────────────►
```

### 11.4 行分批双缓冲 pingpong
```
rowLoopIdx=0: pingpongFlag=0 → sUbOffset=0
  CopyS → ls          ScaleS/compute在slot0     SubCoreCompute on slot0
rowLoopIdx=1: pingpongFlag=1 → sUbOffset=16384
  WaitFlag(V_MTE2)    CopyS next → ls          (slot0计算中ls已空闲)
                       SetFlag(MTE2_V)
                       WaitFlag → ScaleS在slot1
                                             SubCoreCompute on slot1
rowLoopIdx=2: pingpongFlag=0 → sUbOffset=0 (slot0已被MTE3写回GM后可复用)
  ...
```

## 12. 设计亮点与注意事项

### 设计亮点
1. **数值稳定的 online softmax**：通过维护 m/l 跨 tile 累积量，避免了整体 softmax 的 HBM 读写，数值精度与全量 softmax 等价。
2. **maskUbTensor16 复用 LS 区**：S 搬到 compute 区后，LS 着陆区即可被 half mask 复用，节省 16KB UB。
3. **512列快速路径**：`RowsumSPECTILE512` 用3次Add+1次WholeReduceSum替代通用路径的for循环，减少PipeBarrier开销。
4. **子核二维拆分**：decode 场景按序列维拆，prefill 场景按 head 维拆，在不同场景下都实现负载均衡。
5. **mask 三段加载**：prologue/integral/epilogue 三段 DataCopyPad 处理多头并行时的行对齐问题。
6. **低精度节省UB**：half 版本 UB 占用小于 float 版本，为后续功能扩展或更大 tile 留出空间。

### 注意事项
1. **当前未实例化**：代码中所有 kernel 启动均用 `IntermCalcPrec=float`，本文件是预留路径，若要启用需在 flash_api.cpp 增加 `IntermCalcPrec=half` 的模板实例并做精度验证。
2. **FP16 精度风险**：half 尾数仅 10bit，行规约和 dm/l 递推在长序列(>2K)场景可能有精度损失；高精度 float 版本是当前安全选择。
3. **-6e4 而非 -inf**：直接填 -inf 可能在某些 Vector 指令上产生 NaN，-6e4 接近 half 最小值(-65504)，softmax 后下溢到 0 且不会产生 NaN。
4. **scaleValue 仍在 Vector 侧**：Cube MMAD 输出不融合 scale，保持与高精度版本一致的软件架构，便于统一模板。
5. **isLastNoMaskStackTile 未实现**：相比高版本，缺少最后无mask块的MTE3_MTE2排空同步，可能导致尾块流水线不严谨；启用前需要补齐。
6. **preLoad 未实现**：高版本通过 `rowLoopIdx < rowLoopNum + 1` 实现 DMA 预取，本版本串行，带宽利用率较低。
7. **softcap 不支持**：若未来需要支持 Gemma2 等带 softcap 的模型，必须使用 float 版本或补充 half softcap 实现。

## 13. 总结

`online_softmax_low_prec.hpp` 实现了 FP16 精度的分块 online softmax Vector epilogue，承担 FlashAttention 前向推理中从 S=QK^T 到 P=softmax(S) 的计算与写回。它展示了 Ascend C 编程中的典型技巧：

- **UB 静态分区 + 地址复用**（mask16 复用 LS 区）
- **跨 tile 累积量 m/l 的在线更新**（online softmax 核心）
- **Vector 指令的谓词掩码处理尾块**（SetVecMask）
- **Brcb 广播 + Sub/Exp/Mul/Add 实现逐行 softmax**
- **子核级任务拆分**（序列维 vs head 维）
- **MTE2/V/MTE3 多 pipe 软流水**（EVENT_ID0/1/3 乒乓）
- **causal mask 的分块加载与偏移应用**

作为预留路径，它与高精度 float 版本共享接口但用 half 获得更小的 UB 占用，是未来性能优化（如混合精度、感知蒸馏场景）的基础设施。
