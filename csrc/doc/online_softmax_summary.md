# online_softmax.hpp 详解

## 1. 文件定位

`online_softmax.hpp` 是 FlashAttention NPU 前向推理 pipeline 中 **Vector 核 FP32 高精度 Online Softmax epilogue** 的 CATLASS BlockEpilogue 偏特化实现，也是当前所有 kernel 实例化**实际使用**的版本。

- **物理路径**：`csrc/flash_attn_npu/online_softmax.hpp`
- **命名空间**：`Catlass::Epilogue::Block`
- **偏特化签名**：`BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>, ...>`
- **对应低精度版本**：[online_softmax_low_prec.hpp](file:///d:/jianshuqiang/code/flash-attention-npu/csrc/flash_attn_npu/online_softmax_low_prec.hpp)（`SM_DTYPE_=half`，预留未启用）
- **启用状态**：✅ **当前启用** —— `flash_api.cpp` 中所有 `IntermCalcPrec=float` 的 kernel 实例均使用本文件
- **下游 epilogue**：计算完成后由 `rescale_o.hpp` 做最终 O 归一化
- **相关文件**：[fa_block.h](file:///d:/jianshuqiang/code/flash-attention-npu/csrc/flash_attn_npu/fa_block.h) 定义 `EpilogueAtlasA2OnlineSoftmaxT` 模板和 `LseModeT` 枚举

## 2. 在前向 Pipeline 中的位置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           单个 AI Core (Atlas A2)                          │
│                                                                             │
│  ┌──────────────┐  qkReady  ┌────────────────────────────────┐ softmaxReady │
│  │  Cube Core   │──────────►│      Vector Core (×2 sub-core) ├──────────────►── PV matmul
│  │              │           │                                │              │
│  │ BlockMmadQK  │           │ 本文件 (FP32, 实际使用版本):    │              │
│  │ Q×K^T → S    │           │  ① CopyS GM→UB                 │              │
│  │              │           │  ② ScaleS (×1/√d)             │              │
│  │              │           │  ③ ApplySoftcap (可选)         │              │
│  │              │           │  ④ CopyMask/UpCast/ApplyMask   │              │
│  │              │           │  ⑤ OnlineSoftmax 6步(m/l/P)    │              │
│  │              │           │  ⑥ DownCastP (float→half/bf16) │              │
│  │              │           │  ⑦ CopyPUbToGm (P→GM)         │              │
│  └──────────────┘           └────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**跨核同步链**：Cube `CrossCoreSetFlag(qkReady=ID1)` → Vector 等待后执行 softmax → Vector `CrossCoreSetFlag(softmaxReady=ID2)` → Cube 开始 PV matmul。

## 3. Online Softmax 数学原理

FlashAttention 将 K/V 沿序列维切成大小为 `MAX_KV_STACK_LEN=512` 的 tile，逐 tile 流式计算 softmax。核心是维护两个跨 tile 的累积统计量实现**分块数值稳定 softmax**：

### 3.1 三个核心量

| 符号 | 数学定义 | 物理含义 | UB存储 |
|------|----------|----------|--------|
| **m_t** | `m_t = max(m_{t-1}, rowmax(S_t))` | 全局行最大值（防止exp上溢） | `gmUbTensor` (m_old) / `hmUbTensor` (m_new) |
| **l_t** | `l_t = exp(m_{t-1}-m_t)·l_{t-1} + rowsum(exp(S_t - m_t))` | 全局归一化分母 | `glUbTensor` |
| **P_t** | `P_t = exp(S_t - m_t)` （**未归一化**） | 当前 tile 的概率矩阵 | GM `gP`（给PV用） |

### 3.2 为什么 P 不除以 l_t？

P_t = exp(S_t - m_new) 没有除以 l_new，这是 FlashAttention 的关键洞察：
- PV matmul 计算 O_t = P_t × V_t（未归一化加权 V）
- rescale_o epilogue 中用 `O_new = exp(m_old-m_new)·O_old + O_t` 延迟合并
- 最终除以 l_final 才完成归一化
- 好处：避免每 tile 除法，P 直接作为 fp16/bf16 写回节省带宽

### 3.3 七步计算流（SubCoreCompute<doTriUMask> 内部）

```
① CalcLocalRowMax        : m_local = rowmax(S_t)                          → lm
② UpdateGlobalRowMax     : m_new = max(m_local, m_old)                     → hm
                           dm     = exp(m_old - m_new)                     → dm
                           m_old ← m_new                                   → gm
③ CalcExp                : P_t = exp(S - m_new)（含Brcb广播）              → ls
④ [Wait MTE3_V]          : 等前批P写回完成（仅无mask路径）
⑤ DownCastP              : float → half(CAST_NONE) / bf16(CAST_RINT)      → lp
   [SetFlag V_MTE3]
⑥ CalcLocalRowSum        : l_local = rowsum(P_t)                           → ll
   [SetFlag V_MTE2]
   [Wait V_MTE3]
⑦ CopyPUbToGm            : P → GM(gP)
   [根据doTriUMask设置不同排空Flag]
⑧ UpdateGlobalRowSum     : l_new = dm·l_old + l_local                      → gl
```

## 4. FP32 版本 vs FP16 版本 11 项关键差异

| 维度 | online_softmax.hpp (FP32，本文件) | online_softmax_low_prec.hpp (FP16) |
|------|----------------------------------|-----------------------------------|
| **启用状态** | ✅ 当前所有kernel实际使用 | ❌ 预留未实例化 |
| **中间精度** | float (FP32) | half (FP16) |
| **S 计算缓冲** | **原地**在 lsUbTensor[sUbOffset]，无独立 compute 区 | 需要独立 computeUbTensor |
| **S 区 UB 容量** | 8192 float = 32KB | 16384 half = 32KB（相同字节） |
| **lp/mask/mask32 布局** | **三者时间复用** 64KB 起始偏移 | lp在64KB、mask在176KB、mask16复用LS区 |
| **mask16 位置** | 独立 176KB（不与LS复用） | 复用LS区0偏移（S已搬走） |
| **mask 填值** | (float)-3e38（接近float负极限） | (half)-6e4 |
| **行规约算法** | **三级级联** BlockReduceSum/Max；有 SPECTILE512/SPECTILE256/TAILTILE 三路径 | Add+WholeReduceSum 两级；仅 SPECTILE512+TAILTILE |
| **softcap 支持** | ✅ ApplySoftmax 实现 `c·tanh(x)=2c/(1+e^{-2x})-c` | ❌ softcapValue_参数被注释 |
| **preLoad 预取流水线** | ✅ preLoad=1（DMA与计算重叠1个迭代） | ❌ 串行执行 |
| **P 输出** | 需 DownCastP(float→half/bf16) | 直接 DataCopy（同类型） |
| **SubCoreCompute** | 模板 `<bool doTriUMask>`，事件ID/排空同步不同 | 非模板，逻辑简化 |
| **isLastNoMaskStackTile** | ✅ 用于MTE3_MTE2排空同步 | 接收但未使用 |
| **row0 mask预加载** | ✅ 在 CrossCoreWaitFlag(qkReady) **之前**发起mask DMA，隐藏等待延迟 | 在CrossCoreWaitFlag之后加载mask |
| **SetVecMask** | 按位循环构造（float64元素/向量） | 分支构造（half128元素/向量） |
| **CopyS DMA 粒度** | FLOAT_BLOCK_SIZE=8 (float元素) | BLOCK_SIZE=16 (half元素) |

## 5. UB 内存布局

```
 字节偏移  大小     张量                 用途与复用关系
────────────────────────────────────────────────────────────────────────
 0KB       64KB     lsUbTensor           S输入+计算 (float, pingpong双槽各32KB)
                          buffer0: [0KB,32KB), buffer1: [32KB,64KB)
                          scale/softcap/sub/exp 均原地操作
 64KB      ~96KB    lpUbTensor           P输出 (half/bf16, 约32KB)
                   maskUbTensor         int8 mask GM→UB着陆 (≤16KB)  ─┐
                   maskUbTensor32       float mask UpCast后 (≤32KB) ──┴─ 时间复用
                                                                    (不同阶段活跃)
 160KB     8KB      tvUbTensor           Brcb广播/BlockReduce临时 (float)
 168KB     1KB      lmUbTensor           m_local (当前tile行max)
 169KB     1KB      hmUbTensor           m_new = max(lm, gm)
 170KB     1KB      gmUbTensor           m_old (历史累积行max)
 171KB     1KB      llUbTensor           l_local (当前tile行sum)
 172KB     1KB      glUbTensor           l_new = dm·l_old + ll
 173KB     1KB      dmUbTensor           delta-m = exp(m_old-m_new)
                   (6个行向量共6KB，各1KB=256 floats，支持256行/子核)
 176KB     16KB     maskUbTensor16       half mask 中间缓冲 (独立区域)
 192KB
```

**时间复用详解**（64KB~160KB 区域）：
1. **S 计算阶段**（ScaleS→Softcap→CalcExp）：此区域尚未使用，ls占用0~64KB
2. **Mask 加载阶段**：maskUbTensor(int8) 和 maskUbTensor32(float) 占用此区域
3. **P 输出阶段**（DownCastP→CopyPUbToGm）：lpUbTensor(half/bf16) 占用此区域；此时 mask 已用完、S 已exp完毕不再需要

## 6. 类结构与成员函数

```
BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, float>, ...>
│
├── 类型别名 : DispatchPolicy/ArchTag/Element{Output,Input,Mask}/Layout{...}
├── 常量定义 : BLOCK_SIZE/FLOAT_BLOCK_SIZE/FLOAT_VECTOR_SIZE/HALF_VECTOR_SIZE/
│             MAX_UB_S_ELEM_NUM=8192/REDUCE_UB_SIZE=1024/MAX_ROW_NUM_SUB_CORE=256
│
├── 构造/析构
│   └── BlockEpilogue(resource, scaleValue, softcapValue=0) : 分配所有UB张量
│
├── 工具函数
│   ├── Min<T>(a,b)                     : 两值取小（自定义，避免std::min在AI Core上的问题）
│   ├── SetVecMask(len)                 : float64元素向量谓词掩码（循环按位置1）
│   └── SetBlockReduceMask(len)         : BlockReduce专用8元素掩码
│
├── 行规约（Rowmax/Rowsum 各3路径）
│   ├── {Rowmax,Rowsum}SPECTILE512      : 512列=8个vec，三级BlockReduce级联
│   ├── {Rowmax,Rowsum}SPECTILE256      : 256列=4个vec，两级+掩码规约
│   └── {Rowmax,Rowsum}TAILTILE         : 通用/尾块路径，for循环分块规约+Add/Max累积
│
├── 数据搬运
│   ├── CopySGmToUb                     : S(float): GM→UB(ls), FLOAT_BLOCK_SIZE=8
│   ├── CopyMaskGmToUb                  : mask(int8): GM→UB, 三段prologue/integral/epilogue
│   └── CopyPUbToGm                     : P(half/bf16): lp→GM(gP), BLOCK_SIZE=16
│
├── S 预处理
│   ├── ScaleS                          : S = scaleValue * S (原地float Muls)
│   ├── ApplySoftcap<hasSoftcap>        : c·tanh(x) = 2c/(1+e^{-2x}) - c (6步计算)
│   ├── UpCastMask<Dst,Src>             : int8→half→float 两步Cast
│   └── ApplyMask                       : S += -3e38 * mask32 (整块或部分列)
│
├── Online Softmax 核心计算
│   ├── CalcLocalRowMax                 : 按列数分发到3个Rowmax路径
│   ├── UpdateGlobalRowMax              : 合并m_local和m_old得到m_new和dm
│   ├── CalcExp                         : Brcb(hm)→Sub(S-hm)→Exp(P)
│   ├── CalcLocalRowSum                 : 按列数分发到3个Rowsum路径
│   ├── UpdateGlobalRowSum              : 合并l_local和l_old得到l_new
│   └── DownCastP                       : float→half(CAST_NONE)/bf16(CAST_RINT)
│
├── SubCoreCompute<doTriUMask>          : 单sub-core完整softmax调度（模板分支）
│
└── operator() (两个重载)
    ├── 重载1 (无mask)                 : NO_MASK或完全在对角线下，preLoad=1流水
    └── 重载2 (带causal mask)          : 跨对角线块，row0 mask预加载+二级mask流水
```

## 7. preLoad=1 软流水机制

高版本最关键的优化是 **DMA/Compute 重叠**：循环范围 `[0, rowLoopNum + preLoad)`，将数据搬运和计算分配到不同迭代。

### 7.1 无 mask 版本流水时序

```
时间(迭代) →
rowLoopIdx=0:
  [DMA阶段] CopyS(batch0) → SetFlag(MTE2_V, 0)
  [计算阶段] (不足preLoad=1，跳过)
rowLoopIdx=1:
  [DMA阶段] Wait(V_MTE2,1)→CopyS(batch1)→SetFlag(MTE2_V,1)
  [计算阶段] Wait(MTE2_V,0)→ScaleS(batch0)→Softcap?→SubCoreCompute<false>(batch0)
rowLoopIdx=2:
  [DMA阶段] Wait(V_MTE2,0)→CopyS(batch2)→SetFlag(MTE2_V,0)
  [计算阶段] Wait(MTE2_V,1)→ScaleS(batch1)→...→SubCoreCompute<false>(batch1)
...
rowLoopIdx=rowLoopNum (最后一次):
  [DMA阶段] (rowLoopIdx>=rowLoopNum，跳过)
  [计算阶段] Wait(MTE2_V,pingpong)→ScaleS(最后一批)→SubCoreCompute<false>(最后一批)
```

**效果**：当 SubCoreCompute 计算 batch i 时，DMA 引擎并行搬运 batch i+1 的 S 数据，理论上可将 GM 读带宽隐藏在计算背后。

### 7.2 有 mask 版本额外流水

mask 版本在 preLoad 基础上叠加了**mask 加载流水**：
- **row0 预加载优化**：第一次 mask 拷贝在 `CrossCoreWaitFlag(qkReady)` **之前**执行，利用等 Cube 的时间隐藏 mask DMA 延迟
- **每轮计算阶段末尾**发起下一轮 mask 拷贝（EVENT_ID2 信号），计算和下轮mask搬运并行

```
rowLoopIdx=0 (特殊):
  Wait(MTE3_MTE2, EVENT_ID0) → CopyMask(row0) → SetFlag(MTE2_V, ID2)
  CrossCoreWaitFlag(qkReady)  ← 等待Cube时mask已在DMA
  CopyS(row0) → SetFlag(MTE2_V,0)
rowLoopIdx=1:
  Wait(MTE2_V,ID2) → UpCastMask → mask32就绪
  Wait(MTE2_V,0) → ScaleS(row0) → Softcap → ApplyMask
  CopyMask(row1) → SetFlag(MTE2_V,ID2) [与计算并行]
  SubCoreCompute<true>(row0)
```

## 8. 子核任务拆分

每个 Vector 核有 2 个 sub-core（sub-core 0 和 1），行总数 `rowNum = qSBlockSize × qNBlockSize`，拆分策略：

```
情况A: qNBlockSize == 1 (decode, 单head)
  → 按序列维对半拆
  rowSplitSubBlock = qSBlockSize / 2
  sub-core0: 前半行 [0, qSBlockSize/2)
  sub-core1: 后半行 [qSBlockSize/2, qSBlockSize)
  maskOffsetThisSubBlock = rowOffsetThisSubBlock（mask行偏移不同）

情况B: qNBlockSize > 1 (prefill, 多头并行)
  → 按 head 维拆
  rowSplitSubBlock = qSBlockSize × qNSplitSubBlock
  sub-core0: 前 qNSplitSubBlock 个head的所有行
  sub-core1: 剩余 head 的所有行
  maskOffsetThisSubBlock = 0（mask靠prologue/integral/epilogue处理多头拼接）
```

**行分批大小**：
- `maxRowNumPerLoop = MAX_UB_S_ELEM_NUM(8192) / columnNumRound`
- `rowNumTile = min(RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE=8), FLOAT_VECTOR_SIZE=64)`
- 例：columnNum=512 → maxRowNumPerLoop=16 → rowNumTile=16行/批；columnNum=128 → maxRowNumPerLoop=64 → rowNumTile=64行/批

## 9. 例子1：Prefill + Causal Mask + 512列完整块

### 9.1 场景参数
- batch=1, seqlen=2048, num_heads=32, head_dim=128, dtype=half, causal=true
- qSBlockSize=128, qNBlockSize=1（prefill单头）, MAX_KV_STACK_LEN=512
- scaleValue=1/√128≈0.0884（float）, softcapValue=0（不启用softcap）
- 当前处理 Q 行 [384, 512)，KV tile 列 [384, 512)（对角线穿越块）
- sub-core0 处理行 384~448（64行），sub-core1 处理行 448~512（64行）
- columnNum=128（kvSEndIdx-triUpRoundDown），columnNumRound=RoundUp(128,32)=128（float元素128个=512字节）

### 9.2 mask 区域计算
```
noSkipKvS = 512（行511可见KV≤511）
triUp = noSkipKvS - qSBlockSize = 384
triDown = 512
kvSStartIdx = 384, kvSEndIdx = 512
triUp >= kvSStartIdx → triUpRoundDown = RoundDown(384, 32) = 384
gmOffsetMaskRow = 0, gmOffsetMaskColumn = 0
maskColumn = 512-384 = 128
addMaskUbOffset = 0
```
对角线在 Q行r/KV列=r+1，因此行r∈[384,512) 对KV列[r+1-384,128)做mask。ApplyMask会遍历这些偏移列加-3e38。

### 9.3 行分批
```
maxRowNumPerLoop = 8192 / 128 = 64
rowNumTile = min(RoundDown(64,8), 64) = 64
rowLoopNum = CeilDiv(64,64) = 1（每子核1批处理完）
```

### 9.4 数据流（sub-core 0，64行×128列）
```
preLoad=1, 循环rowLoopIdx=0和1:
rowLoopIdx=0:
  [DMA阶段] Wait(V_MTE2,0) → CopyS(行384~448, 列384~512) → ls[0:64×128=8192 floats=32KB]
            SetFlag(MTE2_V,0)
  Wait(MTE3_MTE2,EVENT_ID0) → CopyMask(row0 mask) → SetFlag(MTE2_V,EVENT_ID2)
  CrossCoreWaitFlag(qkReady)  ← 等Cube完成QK（此时mask在DMA，隐藏延迟）
  [计算阶段] (rowLoopIdx<1, 跳过)
rowLoopIdx=1:
  [DMA阶段] (rowLoopIdx>=rowLoopNum=1, 跳过)
  [计算阶段]
    Wait(MTE2_V,EVENT_ID2) → UpCastMask<half,int8>(mask16) → UpCastMask<float,half>(mask32)
    Wait(MTE2_V,0) → ScaleS(ls[0:8192] *= 0.0884)
    softcapValue=0 → 跳过ApplySoftcap
    CopyMask(row1)→SetFlag(MTE2_V,ID2)（但row1不存在，此处是尾批）
    ApplyMask(maskColumn=128==columnNumRound=128走整块Add)
    SubCoreCompute<true>(batch0):
      CalcLocalRowMax → lm(64 floats)
      UpdateGlobalRowMax(isFirstStackTile=false? 如果之前有KV[0,512)块则false)
        hm=max(lm,gm), dm=exp(gm-hm), gm←hm
      CalcExp: Brcb(hm)→tv, ls=ls-tv, ls=exp(ls)
      DownCastP: float→half CAST_NONE (ls→lp)
      CalcLocalRowSum: columnNum=128 → RowsumTAILTILE
      Wait(V_MTE3) → CopyPUbToGm(lp→gP行384~448,列384~512)
      SetFlag(MTE3_MTE2,EVENT_ID0)
      UpdateGlobalRowSum: gl=dm*gl+ll
```

## 10. 例子2：Decode + GQA + Paged KV Cache + 尾块

### 10.1 场景参数
- Decode阶段：batch=16（16个独立请求），seqlen=1（每batch 1 query）
- GQA配置：qHeads=64, kvHeads=8 → groupSize=8
- qSBlockSize=1（单token），由kernel_common.hpp GetQNBlockTile计算：
  `qNBlockTile = clip((128/1)↓2, [1, groupSize=8])` → min(128,8)=8
- qNBlockSize=8（8个Q head 并行）
- KV cache总长度=300（非对齐，尾块），当前KV tile 列[256,300)（尾块columnNum=44）
- 无causal mask需求（decode时query可见所有KV，因是最后一个token）
- d=64（head_dim=64），scaleValue=1/√64=0.125
- bf16 精度（ElementOutput=bfloat16_t，DownCastP使用CAST_RINT）

### 10.2 子核拆分
```
qNBlockSize=8 > 1 → 按head维拆
qNSplitSubBlock = 8/2 = 4 head/sub-core
rowSplitSubBlock = 1×4 = 4行/sub-core（每行=1token×1head）
sub-core0: 4行（4个Q head）, sub-core1: 4行
```

### 10.3 尾块列数=44
```
columnNum=44, columnNumRound=RoundUp(44, 32B?) Wait
```
注意：这里BLOCK_SIZE是16个half元素，但CopyS用FLOAT_BLOCK_SIZE=8 float=32字节。
```
columnNumRound = RoundUp(44, 16) = 48 (half对齐)
但在CopyS中columnNumRound/FLOAT_BLOCK_SIZE=48/8=6
columnNumPad = layoutInput.stride(0) = 64 (head_dim=64)
```
对于CalcLocalRowMax/CalcLocalRowSum：columnNum=44 < 64(FLOAT_VECTOR_SIZE)
→ 走TAILTILE路径的 `if(numElems<FLOAT_VECTOR_SIZE)` 分支，SetVecMask(44)保护尾元素。

### 10.4 行分批
```
maxRowNumPerLoop = 8192 / 48 ≈ 170.7 → RoundDown(170.7,8)=168 → min(168,64)=64
rowNumTile = 64
rowLoopNum = CeilDiv(4,64) = 1（4行一批）
```

### 10.5 流水时序
```
rowLoopIdx=0:
  [DMA] Wait(V_MTE2,0) → CopyS(4行×44列=176 floats, 到ls[0])
        SetFlag(MTE2_V,0)
  [计算] 跳过 (preLoad=1需要rowLoopIdx>=1才开始计算)
rowLoopIdx=1:
  [DMA] 跳过 (rowLoopIdx>=rowLoopNum=1)
  [计算] Wait(MTE2_V,0) → ScaleS(*0.125)
        softcapValue=0 → 跳过
        SubCoreCompute<false>(batch0, isFirstStackTile=true?):
          LSE_MODE=OUT_ONLY, isFirstStackTile&&isFirstRowLoop → WaitFlag(MTE3_V, EVENT_ID4)
          CalcLocalRowMax(44列) → RowsumTAILTILE+SetVecMask(44) → lm
          UpdateGlobalRowMax(isFirstStackTile=true): hm=lm(DataCopy), dm不计算, gm←hm
          CalcExp: Brcb→Sub(SetVecMask44)→Exp
          WaitFlag(MTE3_V, pingpongFlag=0) → 等前批P写回(首次可能为预灌泡SetFlag)
          DownCastP: bf16走CAST_RINT → lp
          SetFlag(V_MTE3,0)
          CalcLocalRowSum: 44列→TAILTILE→ll
          SetFlag(V_MTE2,0)
          Wait(V_MTE3,0) → CopyPUbToGm(lp→gP,4行×48列,pad=16列)
          isLastNoMaskStackTile? 如果这是最后KV块(300<512)则true
            isLastRowLoop? true(唯一1批)
              WaitFlag(MTE3_MTE2,EVENT_ID0) → SetFlag(MTE3_MTE2,EVENT_ID0)（排空同步）
          UpdateGlobalRowSum(isFirstStackTile=true): gl=ll(DataCopy)
```

## 11. ASCII 图

### 11.1 preLoad=1 DMA/Compute 重叠时序图
```
迭代:   rowLoopIdx=0     rowLoopIdx=1       rowLoopIdx=2       rowLoopIdx=N+1
        ─────────────────────────────────────────────────────────────────
MTE2:   CopyS(b0)        CopyS(b1)          CopyS(b2)          (空闲)
          │                │                  │
          ▼                ▼                  ▼
V(计):  (等待)           ScaleS(b0)         ScaleS(b1)         ScaleS(bN)
                           Softcap?            Softcap?           ...
                           UpCastMask*         UpCastMask*
                           ApplyMask*          ApplyMask*
                           SubCoreCompute(b0)  SubCoreCompute(b1)  SubCoreCompute(bN)
MTE3:                     CopyP(b0)          CopyP(b1)          CopyP(bN)
        ─────────────────────────────────────────────────────────────────
时间→    #####           ###############    ###############    ########
        DMA单独         DMA+Compute并行    DMA+Compute并行     Compute收尾

*: 仅mask重载需要UpCast/ApplyMask
```

### 11.2 三级 BlockReduceSum 级联（512列）
```
srcUb (512 floats/行 = 8 FLOAT_VECTOR × 8 FLOAT_BLOCK):
[ vec0(64f) │ vec1(64f) │ vec2 │ vec3 │ vec4 │ vec5 │ vec6 │ vec7 ]
     │         │         │      │      │      │      │      │
     └─BlockReduceSum(8 blocks→1 block)──┘  (第一级: 8→1 block/行, 存tv)
                          ▼
tv (8 floats/行 = 1 block)
     └─BlockReduceSum(8 blocks→1 block)──┘  (第二级: 跨行? 实际是8→1, 存tv[1024])
                          ▼
tv[REDUCE_UB_SIZE] (1 float/行)
     └─BlockReduceSum→ rowsumUb            (第三级: 最终每行1 float)
                          ▼
                 rowsumUb (l_local, 每行1 float)
```

### 11.3 UB 时间复用示意
```
阶段:    S-scale    S-softcap   mask-load  S-mask   CalcExp    DownCast    CopyP-GM
         │           │                       │        │           │          │
0-64KB:  [===ls===]  [===ls===]  [===ls===]  [==ls==] [===ls===]  [===ls===]  [===ls===]
64-160KB: 空闲        空闲        [=mask/m32] [=m32==]  ...        [==lp===]   [==lp===]
         ls活跃                                                                    
                     ls活跃                  ls活跃    ls活跃      ls活跃后逐渐释放
                                          mask32活跃  P计算前mask用完
                                                               lp活跃(写GM)
```

### 11.4 row0 mask 预加载优化（mask版本）
```
无优化:
  Cube完成QK → CrossCoreWaitFlag(qkReady) → CopyMask → UpCast → ApplyMask → Softmax
                  ^^^^^^^^^^^^^^^^^^^^^                    ^^^^^^^^^^^^^^^^
                  等待时间(空闲)                            mask延迟

本文件优化:
  CopyMask(发起DMA) → CrossCoreWaitFlag(qkReady) → UpCast → ApplyMask → Softmax
  ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^
  mask在DMA            Cube和mask并行完成          mask数据已就位
  (与Cube计算重叠)
```

### 11.5 SubCoreCompute<doTriUMask> 事件同步差异
```
无mask路径 (doTriUMask=false):
  ...CalcExp → WaitFlag(MTE3_V, pingpongFlag=0/1) → DownCastP
    └─ 使用 pingpong 变量作为事件ID（0或1），双槽独立流水
  ...CopyP  → SetFlag(MTE3_V, pingpongFlag)
  if (isLastNoMaskStackTile && isLastRowLoop):
    WaitFlag(MTE3_MTE2, EVENT_ID0) → SetFlag(MTE3_MTE2, EVENT_ID0)
    └─ 最后一个无mask块排空，通知下游Cube/PV可以安全启动

有mask路径 (doTriUMask=true):
  ...CalcExp → (不等待MTE3_V) → DownCastP
    └─ mask路径mask已占用EVENT_ID0做mask DMA同步，不用pingpong事件
  ...CopyP  → SetFlag(MTE3_MTE2, EVENT_ID0)
    └─ 每批都用EVENT_ID0通知mask DMA可复用mask区
```

## 12. 设计亮点与注意事项

### 设计亮点
1. **FP32 高精度计算**：中间 softmax 全 float 精度，避免 FP16 在长序列下的精度损失（尤其行规约累积误差）。
2. **三级 BlockReduce 级联**：SPECTILE512/256 特化路径充分利用硬件 BlockReduce 指令，比通用 Add+WholeReduceSum 更高效。
3. **UB 时间复用**：lp/mask/mask32 共享 64KB 区域，节省约 48KB UB（相比 low_prec 版本的独立布局）。
4. **preLoad=1 软流水**：DMA 与计算重叠，理论上可隐藏大部分 GM 读带宽延迟。
5. **row0 mask 预加载**：在等 Cube 时提前发 mask DMA，关键的延迟隐藏优化。
6. **softcap 编译期裁剪**：`if constexpr (hasSoftcap)` 使非 softcap 模型（如标准 LLaMA/GPT）零开销。
7. **模板化 doTriUMask**：编译期分支消除有/无 mask 路径的运行时开销，事件同步逻辑差异化。
8. **双缓冲 pingpong**：EVENT_ID 按 pingpongFlag=0/1 独立，两槽位 DMA/Compute 并行不冲突。
9. **bf16/half 差异化Cast**：bf16 用 CAST_RINT（banker's rounding）保持数值特性，half 用 CAST_NONE（截断，与half精度匹配）。
10. **MTE3_MTE2 排空同步**：最后一个无 mask tile 做跨 pipe 排空，确保后续 PV matmul 读到完整 P 数据。

### 注意事项
1. **columnNumRound 使用 BLOCK_SIZE=16 对齐而非 FLOAT_BLOCK_SIZE=8**：因为 CopyPUbToGm 目的是 half/bf16，P 矩阵列数按 half BLOCK_SIZE 对齐；但 CopySGmToUb 用 FLOAT_BLOCK_SIZE=8（S 是 float）。这种双重对齐粒度需要小心。
2. **lp/mask/mask32 时间复用的安全性**：依赖执行顺序保证（mask用完后才DownCast，DownCast后才覆盖lp），修改流水时需特别注意。
3. **SetVecMask 逐位构造性能**：循环 `for (i=0;i<temp;i++) mask |= one<<i` 在 len<64 时执行最多63次，但只在尾块调用，开销可忽略。
4. **softcap 公式语义**：代码实现 `c*tanh(x)` 而非 Gemma2 论文中的 `c*tanh(x/c)`，实际使用时scaleValue需与之配合保证语义一致。
5. **isFirstStackTile && isFirstRowLoop 等LSE事件**：EVENT_ID4由外部在灌泡阶段SetFlag，LSE_MODE=NONE时整段被编译期裁剪。
6. **mask版本columnNumRound对齐使用BLOCK_SIZE_IN_BYTE=32**：注意带mask重载中 `RoundUp(columnNum, BLOCK_SIZE_IN_BYTE)` 而重载1用 `RoundUp(columnNum, BLOCK_SIZE=16)`，这是因为mask区域需要按DMA块(32B)对齐。
7. **preLoad循环尾批**：循环上界 `rowLoopNum+preLoad` 需精确处理最后一批的 `delayedRowLoopIdx == rowLoopNum - 1` 判断，确保isLastRowLoop正确。

## 13. 总结

`online_softmax.hpp` 是 FlashAttention NPU 前向 softmax 的**高性能 FP32 实现**，承担从 S=QK^T（已scale）到 P=exp(S-m) 的核心计算与写回。其设计体现了 Ascend C 编程的多项精髓：

- **多粒度 UB 管理**：静态分区+时间复用，兼顾容量和并行度
- **跨 tile online softmax**：m/l 在线更新，数值稳定且节省 HBM
- **三级 BlockReduce 级联**：充分利用硬件规约指令
- **preLoad=1 软流水**：DMA 与 Compute 重叠隐藏带宽延迟
- **跨核/核内多层同步**：CrossCoreFlag(qkReady/softmaxReady)+HardEvent(MTE2/V/MTE3)
- **编译期模板裁剪**：`<bool doTriUMask>`、`<bool hasSoftcap>`、`if constexpr(LSE_MODE)` 零开销分支
- **子核级并行**：2个sub-core按序列/head维拆分任务，负载均衡
- **数据类型精细化**：float中间精度保证数值，half/bf16输出节省带宽，差异化Cast模式

与 low_prec 版本共享接口但内部结构更复杂（三级规约、preLoad流水、mask预加载优化），是生产环境实际使用的成熟实现。
