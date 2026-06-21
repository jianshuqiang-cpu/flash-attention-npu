# rescale_o_low_prec.hpp 详细解析

> **文件位置**: `csrc/flash_attn_npu/rescale_o_low_prec.hpp`
> **调度策略**: `EpilogueAtlasA2RescaleOT<LSE_MODE_, half>`（half = FP16 低精度）
> **实例化**: 当前版本**未被实例化**（预留路径），实际使用 float 版本 `rescale_o.hpp`

---

## 1. 文件定位

本文件是 FlashAttention NPU 前向推理在 **Vector 核**上执行的**最后一个 epilogue**，负责对 PV 矩阵乘输出 OTmp 做跨 KV stack 的**在线重缩放、累加和最终归一化**，产生最终输出 O 和可选的 LSE（log-sum-exp）。

它是 CATLASS 框架 `BlockEpilogue` 模板的特化，运行在 Vector 核上（区别于 QK/PV 运行在 Cube 核上）。

### 流水线位置

```
┌─────────────────────────────────────────────────────────────┐
│  AI Core (含 1 Cube + 2 Vector sub-core)                    │
│                                                             │
│  Cube核:                                                    │
│    QK GEMM → SetFlag(qkReady)                               │
│              PV GEMM (落后2轮) → SetFlag(pvReady)            │
│                                                             │
│  Vector核:                                                  │
│    Wait(qkReady) → online_softmax (softmax+causal mask+P)   │
│                   → SetFlag(softmaxReady)                    │
│    Wait(pvReady) → rescale_o (本文件) ── O重缩放/累加/归一化 │
│                   → 写最终O/GM、可选LSE/GM                   │
└─────────────────────────────────────────────────────────────┘
```

rescale_o 在 CrossCoreWaitFlag(pvReady) 之后执行，此时 Cube 核已完成 PV 矩阵乘，OTmp 已写入 GM workspace。

---

## 2. 核心算法（Online Softmax O 累加）

FlashAttention 的 online softmax 维护三个跨 tile 累加量（由 online_softmax 写入 UB）：

| 量 | 张量 | 含义 |
|---|---|---|
| m_t | `gmUbTensor` (Global Max) | 已处理 tile 的行最大值 |
| l_t | `glUbTensor` (Global Sum of exp) | 已处理 tile 的 exp 归一化和 |
| dm_t | `dmUbTensor` | 当前 tile 的缩放因子 = exp(m_{t-1} - m_t) |

### O 累加公式

当处理第 t 个 KV stack tile（PV 输出为 OTmp_t）：

1. **首块** (t=0, isFirstStackTile=true)：
   ```
   O = OTmp_0
   ```
   直接将 PV 输出作为 O 的初始值。

2. **中间块**：
   ```
   O = O * dm_t + OTmp_t
   ```
   旧 O 乘以缩放因子 `exp(m_{t-1} - m_t)` 后，加上新的 PV 输出。

3. **末块** (t=T-1, isLastStackTile=true)：
   ```
   O = O / l_T
   LSE = ln(l_T) + m_T
   ```
   除以全局 rowsum l_T 完成 softmax 归一化；可选输出 LSE = ln(l) + m。

### 数学推导

标准 softmax 的 O 为：
$$O = \frac{\sum_t \exp(S_t - m_T) \cdot V_t}{\sum_t \exp(S_t - m_T)}$$

通过 online 公式，每步只需要保存 O（分子累加器），无需存储历史 P 或 OTmp：
- 当 m 更新时，旧 O 乘 `exp(m_old - m_new)` 缩放
- 加上新的 `exp(S_new - m_new) * V_new` = OTmp_new
- 最终除以 l（分母）完成归一化

---

## 3. 低精度版 vs 高精度版

| 维度 | 高精度 (rescale_o.hpp) | 低精度 (本文件) |
|------|----------------------|----------------|
| **模板特化** | `EpilogueAtlasA2RescaleOT<LSE_MODE_, float>` | `EpilogueAtlasA2RescaleOT<LSE_MODE_, half>` |
| **中间精度** | FP32 (float) | FP16 (half) |
| **lo/go 类型** | `loUbTensor: float`, `goUbTensor32: float`, `goUbTensor16: ElementOutput` | `loUbTensor: half`, `goUbTensor: ElementOutput`（无float缓冲） |
| **Brcb 重解释** | `ReinterpretCast<uint32_t>()` (float 32位) | `ReinterpretCast<uint16_t>()` (half 16位) |
| **行对齐单位** | `FLOAT_BLOCK_SIZE=8` | `HALF_BLOCK_SIZE=16` |
| **向量宽度** | `FLOAT_VECTOR_SIZE=64` (256B) | `HALF_VECTOR_SIZE=128` (256B) |
| **向量运算类型** | `Mul<half,false>`/`Add<float>`/`Div<float>` → 全float | 全 `half` 精度 |
| **SetMask** | 单 64 位 mask 寄存器（float 64元素） | 双 64 位 mask 寄存器（half 128元素，高/低各64） |
| **末块类型转换** | 需要 `Cast<ElementOutput, float>` 将float→half/bf16 | **无需Cast**，go本身已是half |
| **额外缓冲** | - | `tvUbTensor32: float`（LSE Brcb用）、`lse16/32_ubuf_tensor` |
| **UB复用** | `lse32` 与 `gl` 共享偏移 | `lse16`与`gl`共享，`lse32`与`gm`共享（需多一步half→float Cast） |
| **LSE计算** | Ln/Add 直接float计算，Brcb float→GM | Ln/Add先用half，再Cast<float,half>，Brcb float→GM |
| **实例化状态** | ✅ **实际使用版本** | ❌ 预留（未被flash_api.cpp实例化） |

---

## 4. UB 内存布局

所有 UB 张量通过固定字节偏移从 `ubBuf` 分配，偏移与 `online_softmax_low_prec.hpp` 完全一致，实现 Vector 核 softmax→rescale 的零拷贝数据传递。

```
UB (Unified Buffer) 字节偏移布局（从低地址到高地址）：

偏移(KB)   张量                 大小       用途
────────────────────────────────────────────────────────
0-16      softmax 工作区       16KB      S/P/mask 等（softmax使用）
...
96-112    loUbTensor           16KB      OTmp 加载缓冲（PV输出）
112-128   softmax mask32等     16KB      （softmax使用）
128-160   goUbTensor           32KB      O 累加缓冲(8192个half=16KB)
160-169   tvUbTensor/          ~9KB      Brcb广播临时缓冲
          tvUbTensor32         (~4.5KB)   （half/float复用同一地址）
169-170   hmUbTensor           1KB       预留（softmax使用）
170-172   gmUbTensor/          2KB       m全局最大值（LSE用）
          lse32_ubuf_tensor    2KB       LSE float输出（与gm复用）
172-173   glUbTensor/          1KB       l全局累加和（归一化用）
          lse16_ubuf_tensor    1KB       LSE half中间（与gl复用）
173-...   dmUbTensor           ~1.5KB    dm缩放因子三槽(3×256=768个half=1.5KB)
```

### 时间复用安全

| 缓冲 | 生产者 | 消费者 | 复用时机 |
|------|--------|--------|----------|
| loUbTensor | SubCoreCompute(MTE2) | SubCoreCompute(V) | EVENT_ID3 同步 |
| dmUbTensor | online_softmax | rescale_o | PRE_LAUNCH=2 三槽隔离，读落后2轮 |
| glUbTensor → lse16_ubuf_tensor | online_softmax 写 gl | rescale Brcb/Div 读 gl，之后 Ln 写 lse16 | gl 读完后覆写 |
| gmUbTensor → lse32_ubuf_tensor | online_softmax 写 gm | rescale LSE Add 读 gm，之后 Cast 写 lse32 | gm 读完后覆写 |
| tvUbTensor / tvUbTensor32 | rescale_o 内 Brcb | rescale_o 内 Mul/Div/LSE | 同函数内复用 |

---

## 5. Sub-core 拆分

Atlas A2 每个 AI Core 含 **2 个 Vector 子核**，通过 `GetSubBlockIdx/Num()` 获取索引和总数。rescale_o 支持两种拆分模式：

### Decode 模式（qNBlockSize==1，单Q head单token）
沿**行（序列）维**拆分：
```
总行数 = qSBlockSize (如128)
sub-core0: 行 [0:64)   (inRowSplitSubBlock=64)
sub-core1: 行 [64:128) (inRowActualThisSubBlock=64)
输出写入 gOutput 的对应行偏移
```

### Prefill 模式（qNBlockSize>1，多Q head）
沿**列（head）维**拆分：
```
总列(head)数 = qNBlockSize (如8)
sub-core0: head [0:qNSplitSubBlock) 个head (如0:4)
sub-core1: head [qNSplitSubBlock:qNBlockSize) (如4:8)
输出写入 gOutput 的对应列(head)偏移
每个sub-core处理完整 qSBlockSize 行
```

---

## 6. 行循环与溢出处理

### UB O 容量
`MAX_UB_O_ELEM_NUM=8192` 个 half 元素 = 16KB。
当 `curRowNum * embed > 8192` 时，需要多轮 rowLoop 处理。

对于典型配置 embed=128：
- 每行占用 128 个 half
- UB 可容纳 `8192/128=64` 行
- `rowNumTile = RoundDown(64, 16) = 64` 行

若 curRowNum > 64（如 GQA 多 head 拼接为单 batch），则多轮循环，通过 GM 中的 gUpdate 缓冲区做溢出：
- `needRowLoop=1`
- 非首块：在 go*dm 前先从 gUpdate(GM) 加载上轮 go
- 非末块：在 go=lo+go 后将 go 写回 gUpdate(GM)

```
行循环数据流（2轮，非首块非末块）:

第0轮:
  1. gUpdate(GM) → go (加载上轮保存的O)
  2. OTmp(GM) → lo
  3. dm(Brcb) → tv
  4. go = go * dm
  5. go = lo + go
  6. go → gUpdate(GM) (保存)

第1轮:
  1. gUpdate(GM) → go (加载第0轮保存的O)
  2. OTmp(GM) → lo (下一批行)
  3. dm(Brcb) → tv
  4. go = go * dm
  5. go = lo + go
  6. (末块) go / gl → CopyOToGm
```

---

## 7. 事件同步

rescale_o 使用 6 个 EVENT_ID 进行细粒度流水线同步：

| 事件 | 方向 | 作用 |
|------|------|------|
| EVENT_ID0 | MTE2_V / V_MTE3 | lo/go DMA完成信号、O写回GM排空信号 |
| EVENT_ID1 | MTE2_V | gUpdate→go DMA完成信号（行循环溢出） |
| EVENT_ID3 | V_MTE2 | lo缓冲可覆写信号（SubCoreCompute间传递） |
| EVENT_ID4 | V_MTE3/MTE3_V | LSE写回GM排空/释放信号 |
| EVENT_ID5 | V_MTE3 | go→gUpdate溢出写回排空信号 |
| EVENT_ID6 | MTE3_MTE2 | SubCoreCompute入口排空信号 |

---

## 8. 例子 1：Prefill MHA 单 stack tile 首块

### 配置

| 参数 | 值 |
|------|-----|
| seqlen | 512 |
| d_head (embed) | 128 |
| num_heads | 32 (MHA, qNBlockSize=1) |
| qSBlockSize | 128 |
| rowNum | 128 |
| embed | 128 |
| isFirstStackTile | true |
| isLastStackTile | false（假设有多KV stack）|
| LSE_MODE | NONE(0) |
| subBlockNum | 2 |

### 循环参数

- `maxRowNumPerLoop = 8192/128 = 64`
- `rowNumTile = RoundDown(64, 16) = 64`
- `rowLoop = CeilDiv(128(sub-core分配64行), 64) = 1`
- `needRowLoop = 0`（单轮即可）

Sub-core 拆分（qNBlockSize==1, Decode-like按行拆）:
- sub-core0: `inRowSplitSubBlock=128/2=64`行，`inRowActualThisSubBlock=64`
- sub-core1: `inRowActualThisSubBlock=128-64=64`行

### SubCoreCompute 执行（sub-core0, 64行）

```
isFirstStackTile=true, isLastStackTile=false:
  1. Wait EVENT_ID6（MTE3_MTE2 排空）→ 通过（初始状态）
  2. (跳过非首块分支)
  3. 进入首块分支(else):
     DataCopy gInput(OTmp[0:64,:]) → goUbTensor
       DataCopyParams(1, 64*128/16=512个block, 0, 0)
     Set MTE2_V(EVENT_ID0)
     Wait MTE2_V(EVENT_ID0) → DMA完成
  4. isLastStackTile=false → 跳过末块分支
  5. needRowLoop=0 → 跳过溢出写回
  6. Set MTE3_MTE2(EVENT_ID6) → 标记完成
```

此时 goUbTensor 中存放 OTmp 前64行，作为O的初始值。下一个 KV stack tile 到来时走非首块路径。

---

## 9. 例子 2：Decode GQA 末块归一化+LSE输出

### 配置

| 参数 | 值 |
|------|-----|
| seqlen_q | 1 (Decode) |
| seqlen_kv | 2048 |
| d_head (embed) | 128 |
| num_heads | 32, num_kv_heads=4 (GQA groupSize=8) |
| qSBlockSize | 1 |
| qNBlockSize | 8 (当前group8个Qhead) |
| stackSeqTile | 512 |
| rowNum | 1×8=8行 |
| isFirstStackTile | false（非首块）|
| isLastStackTile | true（最后KV stack）|
| LSE_MODE | OUT_ONLY(1) |
| curStackTileMod | 2 (最后槽位) |

### 循环参数

- `maxRowNumPerLoop=8192/128=64`, `rowNumTile=RoundDown(64,16)=64`
- `rowLoop=CeilDiv(8,64)=1`, `needRowLoop=0`

Sub-core 拆分（qNBlockSize=8>1, Prefill模式按head列拆）:
- `qNSplitSubBlock=8/2=4`
- sub-core0: `qNThisSubBlock=4`个head, 处理完整qSBlockSize=1行
- sub-core1: `qNThisSubBlock=8-4=4`个head
- `inRowSplitSubBlock = qSBlockSize*qNSplitSubBlock = 1*4=4`行
- 但实际 rowNum=8，sub-core0: `inRowActualThisSubBlock=4`, sub-core1: `8-4=4`

### SubCoreCompute 执行（sub-core0，4行=4个Qhead）

```
isFirstStackTile=false, isLastStackTile=true, needRowLoop=0:

A. 非首块加载lo:
  Wait V_MTE2(EVENT_ID3)
  DataCopy gInput(OTmp) → loUbTensor (4×128=512个half, 32个block)
  Set MTE2_V(EVENT_ID0)

B. Wait MTE3_MTE2(EVENT_ID6)

C. go重缩放准备:
  SetVectorMask全1
  Brcb: dm[curStackTileMod*256+rowOffsetLoop] → tvUbTensor
       curRowNumRound = RoundUp(4, 16) = 16
       16/8=2次Brcb块, 每次scalar重复8次
  PipeBarrier<PIPE_V>
  (needRowLoop=0, 跳过gUpdate加载)
  逐向量Mul: go = go * dm
    embed/HALF_VECTOR_SIZE=128/128=1 次Mul, 无尾向量
    curRowNum=4, BinaryRepeatParams(1,1,0,8,8,1)
  PipeBarrier<PIPE_V>
  Wait MTE2_V(EVENT_ID0) → lo DMA完成
  Add: go = lo + go
    (4*128+127)/128 = 4个向量, BinaryRepeatParams(1,1,1,8,8,8)
  PipeBarrier<PIPE_V>
  Set V_MTE2(EVENT_ID3)

D. 末块归一化:
  Brcb: gl[rowOffsetLoop] → tvUbTensor (广播l到行向量)
  PipeBarrier<PIPE_V>
  逐向量Div: go = go / gl (1次Div)
  PipeBarrier<PIPE_V>
  Set V_MTE3(EVENT_ID0); Wait V_MTE3(EVENT_ID0)

E. CopyOToGm（三段式）:
  rowOffsetLoop=0, rowActualCurLoop=4
  qSThisSubBlock=1 (Prefill模式,每head1行)
  proTokenIdx = 0 % 1 = 0
  proTokenNum = min(4, 1-0) % 1 = min(4,1)%1 = 0
  integralHeadNum = (4-0)/1 = 4 个完整head
  epiTokenNum = 4-0-4*1 = 0
  → 4次DataCopyPad, 每次写1行×embed=128个half到gOutput
  → GM行间隔(oHiddenSize-embed)*2B: 跨head stride

F. LSE输出 (isLastRowLoop=true, LSE_MODE=OUT_ONLY):
  PipeBarrier<PIPE_V>
  Ln<half>(lse16, gl) → lse16=ln(l) per row
    CeilDiv(4,128)=1个向量
  PipeBarrier<PIPE_V>
  Add<half>(lse16, lse16, gm) → lse16=ln(l)+m=LSE (half)
  PipeBarrier<PIPE_V>
  Cast<float,half>(lse32, lse16, CAST_NONE) → float LSE
    CeilDiv(4,64)=1个向量
  PipeBarrier<PIPE_V>
  Brcb(tv32, lse32) → 广播float LSE到向量
    CeilDiv(4,8)=0.5→1次Brcb块
  PipeBarrier<PIPE_V>
  Set V_MTE3(EVENT_ID4); Wait V_MTE3(EVENT_ID4)
  qNThisSubBlock=4≠0 → 逐head写LSE:
    4次DataCopyPad, 每次写qSBlockSize=1个float,
    GM间隔(qHeads-1)*4B
  Set MTE3_V(EVENT_ID4)

G. Set MTE3_MTE2(EVENT_ID6) → 完成
```

### 数据流图

```
GM:OTmp[0:4,:] ──MTE2──▶ UB:loUbTensor[0:4,:]
                              │
UB:dm[2*256+0] ──Brcb──▶ tvUbTensor[0:4,:]
                              │
UB:goUbTensor (来自上轮累加) ──Mul──▶ go=go*dm ──Add──▶ go=lo+go
                                                         │
UB:gl[0] ──Brcb──▶ tvUbTensor[0:4,:]                     │
                    ┌────────────────────────────────────┘
                    ▼
                 Div: go=go/gl ──▶ 归一化后O ──MTE3──▶ GM:O[output偏移]
                                              │
UB:gl ──Ln──▶ lse16=ln(l) ──Add(+gm)──▶ lse16=LSE(half)
                                        │
                                   Cast<float,half>
                                        ▼
                                     lse32=LSE(float) ──Brcb──▶ tv32
                                                                   │
                                                              DataCopyPad
                                                                   ▼
                                                             GM:gLse[LSE偏移]
```

---

## 10. 关键设计细节

### Multi-head 三段式输出

行循环中一轮可能跨越多个 head 边界（尤其Prefill模式下多个head拼接成连续行）。`CopyOToGm` 分三段处理：

```
行循环内的行排列（qSThisSubBlock=qSBlockSize=128, 3个head为例）:
  行0-127:  head0 tokens (128行完整head)
  行128-255: head1 tokens (128行完整head)
  行256-383: head2 tokens (128行完整head)

若rowActualCurLoop=300行, rowOffsetLoop=50:
  proTokenIdx = 50%128 = 50
  proTokenNum = min(300, 128-50)%128 = min(300,78)%128 = 78
    → 从行50开始的78个token属于head0的尾部
  integralHeadNum = (300-78)/128 = 1
    → 1个完整head (head1, 128行)
  epiTokenNum = 300-78-1*128 = 94
    → head2的前94个token

写GM:
  prologue:  78行×128half, GM起始=proTokenIdx*oHiddenSize, 行间stride=(oHiddenSize-embed)*2B
  integral:  1次128行×128half, 同理
  epilogue:  94行×128half
```

DataCopyPad 的 `(oHiddenSize-embed)*2B` 参数实现了从UB连续布局到GM多head stride布局的转换——每行连续写embed个half，跳过(oHiddenSize-embed)字节到达下一行的该head位置。

### dm 三缓冲

dm 按 `curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop` 索引：
- `curStackTileMod = (stackSeqCount - PRE_LAUNCH) % 3 = (stackSeqCount-2) % 3`
- 每槽 MAX_ROW_NUM_SUB_CORE=256 行
- PRE_LAUNCH=2 流水线深度确保读落后写2轮，三槽互不冲突

### Brcb 广播机制

Brcb（Broadcast）将 per-row scalar 复制到整行向量：
- `BrcbRepeatParams(1, 8)`：每个scalar在block内重复8次
- 对于half(16位)：8*16=128位=8个half=1个float block？ 实际是一个half scalar被复制到8个连续half位置，满足向量宽度需求
- 对于float(32位)：8*32=256位=8个float，对应FLOAT_BLOCK_SIZE=8

### EVENT_ID6 MTE3_MTE2 入口排空

WaitFlag(MTE3_MTE2, EVENT_ID6) 在 SubCoreCompute 入口处等待，确保上次 SubCoreCompute 的 MTE3 写操作（gLse写回或gUpdate溢出写回）已完成，不会与本次 MTE2 读操作冲突。对应出口处 SetFlag(MTE3_MTE2, EVENT_ID6)。

---

## 11. 设计亮点

1. **Online softmax 一致的 O 累加**：O = O*dm + OTmp 公式与 m/l 维护同步，无需存储历史 P，实现真正的 O(N) memory。
2. **UB 零拷贝共享**：通过固定字节偏移，dm/gl/gm 由 online_softmax 写、rescale_o 读，无需 GM 中转。
3. **缓冲时序复用**：gl/gm 与 lse16/lse32 共享偏移（gl读完后覆写为lse16，gm读完后覆写为lse32），节省UB空间。
4. **双精度支持**：同一算法在rescale_o.hpp(float)和rescale_o_low_prec.hpp(half)两个文件中实现，编译期选择精度。
5. **Sub-core双维度拆分**：Decode沿行、Prefill沿head，充分利用2个Vector子核并行。
6. **行循环溢出机制**：UB容量不足时通过gUpdate(GM)做中转，支持任意行数的O累加。
7. **三段式DataCopyPad输出**：prologue/integral/epilogue处理行循环跨越head边界的非对齐情况，同时完成UB连续→GM strided布局转换。
8. **6个EVENT_ID精细同步**：MTE2/MTE3/V管道各事件独立标记，实现DMA-Compute-DMA三级重叠。
9. **三槽dm隔离**：PRE_LAUNCH=2深度流水线下dm三槽互不冲突，Cube/Vector核间延迟容忍。
10. **LSE可选输出**：通过LSE_MODE模板参数编译期分支，不输出LSE时零开销。

---

## 12. 注意事项

- **本文件为预留版本**：flash_api.cpp中所有kernel实例化均使用IntermCalcPrec=float，对应rescale_o.hpp（高精度版）是实际运行版本。
- **hmUbTensor在rescale中未使用**：它由online_softmax写入（局部row max），rescale_o不需要读（rescale只用dm/gl/gm），但仍分配以保持UB偏移与softmax一致。
- **`proTokenIdxPre`声明但未使用**：operator()中声明了`uint32_t proTokenIdxPre=0`但后续未引用，为预留变量。
- **`qSRemian`声明但未使用**：`uint32_t qSRemian = qSThisSubBlock`未被使用，为预留。
- **Brcb块计数**：`curRowNumRound/FLOAT_BLOCK_SIZE`对half Brcb也除以FLOAT_BLOCK_SIZE(8)而非HALF_BLOCK_SIZE(16)，这是因为Brcb以uint16_t重解释后以32位block计数（8个uint16_t = 128bit = 1个Brcb块？），需结合硬件Brcb指令理解。
- **低精度风险**：half精度下O累加可能产生较大数值误差，特别是长序列场景下dm*go的精度损失，这也是默认使用float版本的原因。
- **lse16/lse32复用时机**：lse16在末块Div完成后才写入gl位置（gl已被Brcb+Div消费），lse32在Add(lse16+gm)完成后才写入gm位置（gm已被Add消费），时序安全。

---

## 13. 总结

`rescale_o_low_prec.hpp` 以约460行代码实现了FlashAttention前向推理最后阶段的Vector核 epilogue，核心职责是：

1. **跨KV stack的O在线累加**：通过 `O = O*dm + OTmp` 公式在half精度下维护O分子累加器；
2. **末块归一化**：除以全局rowsum l得到最终softmax输出O；
3. **可选LSE输出**：`ln(l)+m`计算log-sum-exp，half→float转换后写GM；
4. **Sub-core并行**：Decode沿行/Prefill沿head的两维度拆分；
5. **UB溢出处理**：行循环+gUpdate GM中转支持大行数场景；
6. **三段式输出**：prologue/integral/epilogue处理head边界。

它与online_softmax_low_prec.hpp通过共享UB偏移实现零拷贝数据传递，与pv_matmul.hpp通过CrossCoreWaitFlag(pvReady)跨核同步，共同构成FlashAttention前向推理的完整Vector核计算链路。
