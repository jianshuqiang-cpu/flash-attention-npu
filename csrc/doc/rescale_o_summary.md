# rescale_o.hpp 详细解析

> **文件位置**: `csrc/flash_attn_npu/rescale_o.hpp`
> **调度策略**: `EpilogueAtlasA2RescaleOT<LSE_MODE_, float>`（float = FP32 高精度）
> **实例化**: ✅ **实际使用版本** —— `flash_api.cpp` 中所有 kernel 实例化均使用 `IntermCalcPrec=float`，对应本文件；`rescale_o_low_prec.hpp`（half 版）仅作为预留路径未被实例化。

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

rescale_o 在 `CrossCoreWaitFlag(pvReady)` 之后执行，此时 Cube 核已完成 PV 矩阵乘，OTmp 已写入 GM workspace。

---

## 2. 核心算法（Online Softmax O 累加）

FlashAttention 的 online softmax 维护三个跨 tile 累加量（由 `online_softmax.hpp` 写入 UB）：

| 量 | 张量 | 含义 |
|---|---|---|
| m_t | `gmUbTensor` (Global Max) | 已处理 tile 的行最大值 |
| l_t | `glUbTensor` (Global Sum of exp) | 已处理 tile 的 exp 归一化和 |
| dm_t | `dmUbTensor` | 当前 tile 的缩放因子 = exp(m_{t-1} - m_t) |

### O 累加公式

当处理第 t 个 KV stack tile（PV 输出为 OTmp_t）：

1. **首块** (t=0, isFirstStackTile=true)：
   ```
   O = OTmp_0          （float 精度直接复制）
   ```
   直接将 PV 输出作为 O 的初始值。

2. **中间块**：
   ```
   O = O * dm_t + OTmp_t    （float 精度 Mul + Add）
   ```
   旧 O 乘以缩放因子 `exp(m_{t-1} - m_t)` 后，加上新的 PV 输出。

3. **末块** (t=T-1, isLastStackTile=true)：
   ```
   O = O / l_T              （float Div 完成归一化）
   O_out = Cast<half/bf16>(O)  （float→输出精度，bf16用CAST_RINT，half用CAST_NONE）
   LSE = ln(l_T) + m_T       （可选：float 精度直接计算）
   ```

### 数学推导

标准 softmax 的 O 为：
$$O = \frac{\sum_t \exp(S_t - m_T) \cdot V_t}{\sum_t \exp(S_t - m_T)}$$

通过 online 公式，每步只需要保存 O（分子累加器），无需存储历史 P 或 OTmp：
- 当 m 更新时，旧 O 乘 `exp(m_old - m_new)` 缩放
- 加上新的 `exp(S_new - m_new) * V_new` = OTmp_new
- 最终除以 l（分母）完成归一化

---

## 3. 高精度版（本文件）vs 低精度版

| 维度 | 高精度 (本文件 rescale_o.hpp) | 低精度 (rescale_o_low_prec.hpp) |
|------|----------------------|----------------|
| **模板特化** | `EpilogueAtlasA2RescaleOT<LSE_MODE_, float>` | `EpilogueAtlasA2RescaleOT<LSE_MODE_, half>` |
| **中间精度** | FP32 (float) | FP16 (half) |
| **lo/go 类型** | `loUbTensor: float`, `goUbTensor32: float` + `goUbTensor16: ElementOutput`（双视图共享地址） | `loUbTensor: half`, `goUbTensor: ElementOutput`（单视图，无 float 缓冲） |
| **Brcb 重解释** | `ReinterpretCast<uint32_t>()` (float 32位) | `ReinterpretCast<uint16_t>()` (half 16位) |
| **行对齐单位** | `FLOAT_BLOCK_SIZE=8` | `HALF_BLOCK_SIZE=16` |
| **向量宽度** | `FLOAT_VECTOR_SIZE=64` (256B) | `HALF_VECTOR_SIZE=128` (256B) |
| **向量运算类型** | `Mul<float>`/`Add<float>`/`Div<float>` → 全 float | 全 `half` 精度 |
| **SetMask** | 单 64 位 mask 寄存器（float 64元素） | 双 64 位 mask 寄存器（half 128元素，高/低各64） |
| **末块类型转换** | 需要 `Cast<ElementOutput, float>`：bf16 用 `CAST_RINT`，half 用 `CAST_NONE` | **无需 Cast**，go 本身已是 half |
| **额外缓冲** | 仅 `tvUbTensor: float`、`lse32_ubuf_tensor: float`（与 gl 复用） | `tvUbTensor32: float`、`lse16/lse32_ubuf_tensor`（多套） |
| **UB 复用** | `lse32` 与 `gl` 共享偏移（gl 读完后覆写） | `lse16` 与 `gl` 共享，`lse32` 与 `gm` 共享（需多步 half→float Cast） |
| **LSE 计算** | `Ln<float>`+`Add<float>` 直接 float 计算，Brcb float→GM | `Ln<half>`+`Add<half>` 后需 `Cast<float,half>` 转 float，Brcb float→GM |
| **UB O 容量** | 8192 个 float = 32KB（embed=128 时 maxRow=64 行） | 8192 个 half = 16KB（embed=128 时 maxRow=64 行） |
| **实例化状态** | ✅ **实际使用版本** | ❌ 预留（未被 flash_api.cpp 实例化） |

---

## 4. UB 内存布局

所有 UB 张量通过固定字节偏移从 `ubBuf` 分配，偏移与 `online_softmax.hpp`（float 版）完全一致，实现 Vector 核 softmax→rescale 的零拷贝数据传递。

```
UB (Unified Buffer) 字节偏移布局（从低地址到高地址）：

偏移(KB)   张量                 大小       用途
────────────────────────────────────────────────────────
0-96      softmax 工作区       96KB      P/S/mask 等（online_softmax 使用）
96-128    loUbTensor           32KB      OTmp 加载缓冲（PV 输出, float）
128-160   goUbTensor32/16      32KB      O 累加缓冲
              ↳ goUbTensor32 (float 视图，Mul/Add/Div 用)
              ↳ goUbTensor16 (ElementOutput 视图，Cast 后 DataCopyPad 用)
                两者起始地址相同(128KB)，Cast 原地写回
160-169   tvUbTensor           ~9KB      Brcb 广播临时缓冲（float，dm/gl/lse 共用）
169-170   hmUbTensor           1KB       预留（softmax 局部 max，rescale 未使用）
170-172   gmUbTensor           2KB       m 全局最大值（LSE 计算用, float）
172-173   glUbTensor /         1KB       l 全局累加和（归一化用, float）
          lse32_ubuf_tensor    (1KB)     LSE 输出（float，末块复用 gl 偏移）
173-...   dmUbTensor           ~3KB      dm 缩放因子三槽（3×256=768 个 float = 3KB）
```

### 双视图 go 共享机制（高精度版核心特性）

```
UB 地址 128KB ──────────────────────────────── 160KB (32KB)
            ┌─────────────────────────────────┐
goUbTensor32│ float float float ... float      │  ← Mul/Add/Div 计算视图
(float)     │ 每元素 4B，共 8192 个 float       │
            ├─────────────────────────────────┤
goUbTensor16│ half/bf16 half/bf16 ... (前半)   │  ← Cast 后输出视图
(ElementOut)│ 每元素 2B，Cast 后仅占前 16KB     │    （DataCopyPad 读取）
            └─────────────────────────────────┘

Cast 指令将 float 结果原地转换为 half/bf16 写入同一地址区域：
  - float 占 4B → half/bf16 占 2B，转换后数据只占原空间的前半部分
  - Cast 后通过 goUbTensor16 视图读取，直接 DataCopyPad 写 GM
```

### 时间复用安全

| 缓冲 | 生产者 | 消费者 | 复用时机 |
|------|--------|--------|----------|
| loUbTensor | SubCoreCompute(MTE2) | SubCoreCompute(V) | EVENT_ID3 同步 |
| dmUbTensor | online_softmax | rescale_o | PRE_LAUNCH=2 三槽隔离，读落后2轮 |
| glUbTensor → lse32_ubuf_tensor | online_softmax 写 gl | rescale Brcb/Div 读 gl，之后 Ln 写 lse32 | gl 读完后覆写（末块 Div 完成后） |
| gmUbTensor | online_softmax 写 gm | rescale LSE Add 读 gm | gm 在末块 LSE Add 时消费，无覆写冲突 |
| tvUbTensor | rescale_o 内 Brcb | rescale_o 内 Mul/Div/LSE | 同函数内复用，PipeBarrier 隔离 |
| goUbTensor32 → goUbTensor16 | Mul/Add/Div 写 go32 | Cast 写 go16，DataCopyPad 读 go16 | Cast 原地转换，V_MTE3 EVENT_ID0 排空后写 GM |

---

## 5. Sub-core 拆分

Atlas A2 每个 AI Core 含 **2 个 Vector 子核**，通过 `GetSubBlockIdx/Num()` 获取索引和总数。rescale_o 支持两种拆分模式：

### Decode 模式（qNBlockSize==1，单 Q head 单 token）
沿**行（序列）维**拆分：
```
总行数 = qSBlockSize (如128)
sub-core0: 行 [0:64)   (inRowSplitSubBlock=64)
sub-core1: 行 [64:128) (inRowActualThisSubBlock=64)
输出写入 gOutput 的对应行偏移（outRowOffsetThisSubBlock=subBlockIdx*64）
```

### Prefill 模式（qNBlockSize>1，多 Q head）
沿**列（head）维**拆分：
```
总列(head)数 = qNBlockSize (如8)
sub-core0: head [0:qNSplitSubBlock) 个 head (如0:4)
sub-core1: head [qNSplitSubBlock:qNBlockSize) (如4:8)
输出写入 gOutput 的对应列(head)偏移（outColOffsetThisSubBlock=subBlockIdx*4*embed）
每个 sub-core 处理完整 qSBlockSize 行
```

---

## 6. 行循环与溢出处理

### UB O 容量
`MAX_UB_O_ELEM_NUM=8192` 个 float 元素 = 32KB。
当 `curRowNum * embed > 8192` 时，需要多轮 rowLoop 处理。

对于典型配置 embed=128：
- 每行占用 128 个 float = 512B
- UB 可容纳 `8192/128=64` 行
- `rowNumTile = RoundDown(64, FLOAT_BLOCK_SIZE=8) = 64` 行

若 curRowNum > 64（如 GQA 多 head 拼接为单 batch），则多轮循环，通过 GM 中的 gUpdate 缓冲区做溢出：
- `needRowLoop=1`
- 非首块：在 go*dm 前先从 gUpdate(GM) 加载上轮 go（EVENT_ID1 同步）
- 非末块：在 go=lo+go 后将 go 写回 gUpdate(GM)（EVENT_ID5 排空）

```
行循环数据流（2轮，非首块非末块，float 精度）:

第0轮:
  1. gUpdate(GM,float) → goUbTensor32 (加载上轮保存的O, MTE2)
  2. OTmp(GM,float) → loUbTensor (加载PV输出, MTE2)
  3. dm[curStackTileMod*256+0] --Brcb(uint32)--> tvUbTensor
  4. go = go * dm  (逐64 float Mul)
  5. go = lo + go  (float Add)
  6. goUbTensor32 → gUpdate(GM,float) (保存, MTE3)

第1轮:
  1. gUpdate(GM,float) → goUbTensor32 (加载第0轮保存的O)
  2. OTmp(GM,float) → loUbTensor (下一批行)
  3. dm[curStackTileMod*256+rowOffset] --Brcb--> tvUbTensor
  4. go = go * dm
  5. go = lo + go
  6. (末块) gl[rowOffset] --Brcb--> tvUbTensor → go=go/gl (Div)
     → Cast<float→half/bf16>(go16=go32) → CopyOToGm → (可选)LSE
```

---

## 7. 事件同步

rescale_o 使用 6 个 EVENT_ID 进行细粒度流水线同步：

| 事件 | 方向 | 作用 |
|------|------|------|
| EVENT_ID0 | MTE2_V / V_MTE3 | lo DMA 完成信号；Cast 后 O 可写 GM 的排空信号 |
| EVENT_ID1 | MTE2_V | gUpdate→go DMA 完成信号（行循环溢出加载） |
| EVENT_ID3 | V_MTE2 | lo 缓冲可覆写信号（SubCoreCompute 间传递） |
| EVENT_ID4 | V_MTE3 / MTE3_V | LSE 写回 GM 排空/释放信号 |
| EVENT_ID5 | V_MTE3 | go→gUpdate 溢出写回排空信号（非末块） |
| EVENT_ID6 | MTE3_MTE2 | SubCoreCompute 入口排空信号（防止 MTE3 写与 MTE2 读冲突） |

---

## 8. 例子 1：Prefill MHA 单 stack tile 首块（float 精度）

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
| isLastStackTile | false（假设有多 KV stack）|
| LSE_MODE | NONE(0) |
| subBlockNum | 2 |

### 循环参数

- `maxRowNumPerLoop = 8192/128 = 64`
- `rowNumTile = RoundDown(64, FLOAT_BLOCK_SIZE=8) = 64`
- `rowLoop = CeilDiv(128(sub-core 分配 64 行), 64) = 1`
- `needRowLoop = 0`（单轮即可）

Sub-core 拆分（qNBlockSize==1, Decode 按行拆）:
- sub-core0: `inRowSplitSubBlock=128/2=64` 行，`inRowActualThisSubBlock=64`
- sub-core1: `inRowActualThisSubBlock=128-64=64` 行

### SubCoreCompute 执行（sub-core0, 64行）

```
isFirstStackTile=true, isLastStackTile=false:
  1. Wait EVENT_ID6（MTE3_MTE2 排空）→ 通过（初始状态）

  2. 进入首块分支(else):
     DataCopy gInput(OTmp[0:64,:], float) → goUbTensor32
       DataCopyParams(1, 64*128/8=1024 个 block, 0, 0)
       （float 元素数 64*128=8192，按 FLOAT_BLOCK_SIZE=8 计 block 数）
     Set MTE2_V(EVENT_ID0)
     Wait MTE2_V(EVENT_ID0) → DMA 完成，go32 就绪

  3. isLastStackTile=false → 跳过末块分支
  4. needRowLoop=0 → 跳过溢出写回

  5. Set MTE3_MTE2(EVENT_ID6) → 标记本轮完成
```

此时 goUbTensor32 中存放 OTmp 前 64 行（float），作为 O 的初始值。下一个 KV stack tile 到来时走非首块路径（go=go*dm+lo）。

---

## 9. 例子 2：Decode GQA 末块归一化+LSE输出（float 精度，bf16 输出）

### 配置

| 参数 | 值 |
|------|-----|
| seqlen_q | 1 (Decode) |
| seqlen_kv | 2048 |
| d_head (embed) | 128 |
| num_heads | 32, num_kv_heads=4 (GQA groupSize=8) |
| qSBlockSize | 1 |
| qNBlockSize | 8 (当前 group 8 个 Q head) |
| stackSeqTile | 512 |
| rowNum | 1×8=8 行 |
| ElementOutput | bfloat16_t（CAST_RINT）|
| isFirstStackTile | false（非首块）|
| isLastStackTile | true（最后 KV stack）|
| LSE_MODE | OUT_ONLY(1) |
| curStackTileMod | 2 (最后槽位) |

### 循环参数

- `maxRowNumPerLoop=8192/128=64`, `rowNumTile=RoundDown(64,8)=64`
- `rowLoop=CeilDiv(8,64)=1`, `needRowLoop=0`

Sub-core 拆分（qNBlockSize=8>1, Prefill 模式按 head 列拆）:
- `qNSplitSubBlock=8/2=4`
- sub-core0: `qNThisSubBlock=4` 个 head, 处理完整 qSBlockSize=1 行
- sub-core1: `qNThisSubBlock=8-4=4` 个 head
- `inRowSplitSubBlock = qSBlockSize*qNSplitSubBlock = 1*4=4` 行
- 实际 rowNum=8，sub-core0: `inRowActualThisSubBlock=4`, sub-core1: `8-4=4`

### SubCoreCompute 执行（sub-core0，4行=4个Qhead）

```
isFirstStackTile=false, isLastStackTile=true, needRowLoop=0:

A. 非首块加载 lo:
  Wait V_MTE2(EVENT_ID3)
  DataCopy gInput(OTmp, float) → loUbTensor (4×128=512 个 float = 64 个 block)
  Set MTE2_V(EVENT_ID0)

B. Wait MTE3_MTE2(EVENT_ID6)

C. go 重缩放准备 (float 精度):
  SetVectorMask 全1 ((uint64_t)-1, (uint64_t)-1)
  Brcb<uint32_t>: dm[2*256+0] → tvUbTensor (float scalar→整行广播)
       curRowNumRound = RoundUp(4, FLOAT_BLOCK_SIZE=8) = 8
       8/8=1 次 Brcb 块, BrcbRepeatParams(1,8): 每 scalar 重复 8 次 = 8 float
  PipeBarrier<PIPE_V>
  (needRowLoop=0, 跳过 gUpdate 加载)
  逐向量 Mul<float>: go = go * dm
    embed/FLOAT_VECTOR_SIZE=128/64=2 次 Mul, 无尾向量
    curRowNum=4, BinaryRepeatParams(1,1,0, 16,16,1)  (embedRound/FLOAT_BLOCK_SIZE=128/8=16)
  PipeBarrier<PIPE_V>
  Wait MTE2_V(EVENT_ID0) → lo DMA 完成
  Add<float>: go = lo + go
    (4*128+63)/64 = 8 个向量, BinaryRepeatParams(1,1,1, 8,8,8)
  PipeBarrier<PIPE_V>
  Set V_MTE2(EVENT_ID3)

D. 末块归一化 (float Div):
  Brcb<uint32_t>: gl[0] → tvUbTensor (广播 l 到行向量)
       8/8=1 次 Brcb 块
  PipeBarrier<PIPE_V>
  逐向量 Div<float>: go = go / gl (2 次 Div, 无尾向量)
  PipeBarrier<PIPE_V>

E. float→bf16 Cast (原地写回 go16):
  Cast<bfloat16_t, float, false>(goUbTensor16, goUbTensor32, CAST_RINT, ...)
       (4*128+63)/64 = 8 个向量
       UnaryRepeatParams(1,1,4,8)
       ⭐ bf16 用 CAST_RINT（就近舍入到偶数），保证数值精度
  Set V_MTE3(EVENT_ID0); Wait V_MTE3(EVENT_ID0) → Vector 管道排空

F. CopyOToGm（三段式，用 goUbTensor16）:
  rowOffsetLoop=0, rowActualCurLoop=4
  qSThisSubBlock=1 (Prefill 模式, 每 head 1 行)
  proTokenIdx = 0 % 1 = 0
  proTokenNum = min(4, 1-0) % 1 = min(4,1)%1 = 0
  integralHeadNum = (4-0)/1 = 4 个完整 head
  epiTokenNum = 4-0-4*1 = 0
  → 4 次 DataCopyPad, 每次写 1 行×embed=128 个 bf16 到 gOutput
  → GM 行间隔 (oHiddenSize-embed)*SIZE_OF_16BIT=(oHiddenSize-128)*2B: 跨 head stride

G. LSE 输出 (isLastRowLoop=true, LSE_MODE=OUT_ONLY, float 直接计算):
  PipeBarrier<PIPE_V>
  Ln<float>(lse32, gl) → lse32=ln(l) per row  （复用 gl 偏移为 lse32）
    CeilDiv(4, FLOAT_VECTOR_SIZE=64)=1 个向量
    UnaryRepeatParams(1,1,8,8)
  PipeBarrier<PIPE_V>
  Add<float>(lse32, lse32, gm) → lse32=ln(l)+m=LSE  （全 float，无需额外 Cast）
    CeilDiv(4,64)=1 个向量
    BinaryRepeatParams(1,1,1,8,8,8)
  PipeBarrier<PIPE_V>
  Brcb<uint32_t>(tv, lse32) → 广播 float LSE 到向量
    CeilDiv(4, FLOAT_BLOCK_SIZE=8)=1 次 Brcb 块
  PipeBarrier<PIPE_V>
  Set V_MTE3(EVENT_ID4); Wait V_MTE3(EVENT_ID4)
  qNThisSubBlock=4≠0 → 逐 head 写 LSE:
    4 次 DataCopyPad, 每次写 qSBlockSize=1 个 float,
    GM 间隔 (qHeads-1)*sizeof(float)=(qHeads-1)*4B
  Set MTE3_V(EVENT_ID4)

H. Set MTE3_MTE2(EVENT_ID6) → 完成
```

### 数据流图

```
GM:OTmp[0:4,:] (float) ──MTE2──▶ UB:loUbTensor[0:4,:] (float)
                                          │
UB:dm[2*256+0] (float) ──Brcb(uint32)──▶ tvUbTensor[0:4,:] (float)
                                          │
UB:goUbTensor32 (上轮累加, float) ──Mul<float>──▶ go=go*dm (float)
                                                                │
                                           Add<float> ◀── lo ───┘
                                                │
                                                ▼
                                          go=lo+go (float)
                                                │
UB:gl[0] (float) ──Brcb(uint32)──▶ tvUbTensor[0:4,:] (float)    │
                                          │                     │
                                   Div<float>: go=go/gl ────────┘
                                          │
                                          ▼
                                  goUbTensor32 (归一化后O, float)
                                          │
                                Cast<float→bf16, CAST_RINT>  ← 就近舍入到偶数
                                          │ (原地写回同地址)
                                          ▼
                                  goUbTensor16 (bf16 视图)
                                          │ MTE3 (DataCopyPad, 三段式)
                                          ▼
                                  GM:gOutput[output 偏移] (bf16)

                                          ┌─── Ln<float>(lse32, gl) ──▶ lse32=ln(l)
UB:gl (float, 已被 Div 消费) ─────────────┤
                                          └─── Add<float>(lse32, lse32, gm) ──▶ lse32=LSE
                                                                                │
UB:gm (float) ────────────────────────────────────────────────────────────────┘
                                                                                │
                                                                    Brcb<uint32>(tv, lse32)
                                                                                │
                                                                          DataCopyPad
                                                                                ▼
                                                                        GM:gLse[LSE偏移] (float)
```

---

## 10. 关键设计细节

### Multi-head 三段式输出

行循环中一轮可能跨越多个 head 边界（尤其 Prefill 模式下多个 head 拼接成连续行）。`CopyOToGm` 分三段处理：

```
行循环内的行排列（qSThisSubBlock=qSBlockSize=128, 3个head为例）:
  行0-127:  head0 tokens (128行完整head)
  行128-255: head1 tokens (128行完整head)
  行256-383: head2 tokens (128行完整head)

若 rowActualCurLoop=300行, rowOffsetLoop=50:
  proTokenIdx = 50%128 = 50
  proTokenNum = min(300, 128-50)%128 = min(300,78)%128 = 78
    → 从行50开始的78个token属于head0的尾部
  integralHeadNum = (300-78)/128 = 1
    → 1个完整head (head1, 128行)
  epiTokenNum = 300-78-1*128 = 94
    → head2的前94个token

写GM (用 goUbTensor16, half/bf16 视图):
  prologue:  78行×128个half/bf16, GM起始=proTokenIdx*oHiddenSize, 行间stride=(oHiddenSize-embed)*SIZE_OF_16BIT
  integral:  1次128行×128个half/bf16, 同理
  epilogue:  94行×128个half/bf16
```

DataCopyPad 的 `(oHiddenSize-embed)*SIZE_OF_16BIT` 参数实现了从 UB 连续布局到 GM 多 head stride 布局的转换——每行连续写 embed 个 half/bf16，跳过 (oHiddenSize-embed) 个元素到达下一行的该 head 位置。

### dm 三缓冲

dm 按 `curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop` 索引：
- `curStackTileMod = (stackSeqCount - PRE_LAUNCH) % 3 = (stackSeqCount-2) % 3`
- 每槽 `MAX_ROW_NUM_SUB_CORE=256` 行（float）
- PRE_LAUNCH=2 流水线深度确保读落后写 2 轮，三槽互不冲突

### Brcb 广播机制（float 版）

Brcb（Broadcast）将 per-row scalar 复制到整行向量：
- `BrcbRepeatParams(1, 8)`：1 个 block 重复，每个 float scalar 在 block 内重复 8 次
- 8 个 float × 32bit = 256bit = 1 个 Brcb 块，对应 `FLOAT_BLOCK_SIZE=8`
- float 版本以 `uint32_t` 重解释（32位），与 half 版本的 `uint16_t`（16位）形成对比
- 块计数 `curRowNumRound / FLOAT_BLOCK_SIZE`，`curRowNumRound = RoundUp(curRowNum, 8)`

### bf16 CAST_RINT vs half CAST_NONE

末块归一化后 float→输出精度 Cast 是高精度版独有的步骤：
- **bf16 输出**：`CAST_RINT`（round to nearest even，就近舍入到偶数）
  - bf16 只有 8 位尾数（vs float 23 位），精度损失大，需四舍五入到偶数减小误差
- **half 输出**：`CAST_NONE`（直接截断）
  - half 有 10 位尾数，精度相对较高，截断即可

`UnaryRepeatParams(1, 1, 4, 8)`：srcRepeatStride=4（float block 步长），dstRepeatStride=8（half/bf16 block 步长），因 half/bf16 元素数是 float 的 2 倍。

### EVENT_ID6 MTE3_MTE2 入口排空

`WaitFlag(MTE3_MTE2, EVENT_ID6)` 在 SubCoreCompute 入口处等待，确保上次 SubCoreCompute 的 MTE3 写操作（gLse 写回或 gUpdate 溢出写回）已完成，不会与本次 MTE2 读操作冲突。对应出口处 `SetFlag(MTE3_MTE2, EVENT_ID6)`。

### SetMask 单 mask 寄存器

float 向量宽度仅 64 个元素（vs half 的 128），使用单个 64 位 mask 寄存器即可覆盖：
- `len == VECTOR_SIZE(128)`：双 mask 全 1（兼容接口）
- `len >= FLOAT_VECTOR_SIZE(64)`：低 64 位全 1，高 64 位按位设置前 (len-64) 个 bit
- `len < 64`：低 64 位按位设置前 len 个 bit，高 64 位全 0

相比 half 版本的双 mask 逻辑，float 版本更简洁。

---

## 11. 设计亮点

1. **实际使用版本**：flash_api.cpp 所有 kernel 实例化均使用 `IntermCalcPrec=float`，本文件是真正运行的高精度 O rescale epilogue。
2. **Online softmax 一致的 O 累加**：`O = O*dm + OTmp` 公式与 m/l 维护同步，无需存储历史 P，实现真正的 O(N) memory。
3. **UB 零拷贝共享**：通过固定字节偏移，dm/gl/gm 由 online_softmax 写、rescale_o 读，无需 GM 中转。
4. **双视图 go 共享**：goUbTensor32(float) 与 goUbTensor16(ElementOutput) 共享同一 UB 地址，Cast 原地写回，省去额外输出缓冲。
5. **缓冲时序复用**：gl 与 lse32 共享偏移（gl 读完后覆写为 lse32），节省 UB 空间；无需 half 版的 lse16 中间缓冲。
6. **LSE 全 float 精度**：Ln/Add 直接 float 计算，无需 half→float Cast 中间步骤，数值精度更高、指令数更少。
7. **bf16/half Cast 分支**：bf16 用 CAST_RINT 保证低精度下的数值准确性，half 用 CAST_NONE 避免不必要开销。
8. **Sub-core 双维度拆分**：Decode 沿行、Prefill 沿 head，充分利用 2 个 Vector 子核并行。
9. **行循环溢出机制**：UB 容量不足时通过 gUpdate(GM) 做中转，支持任意行数的 O 累加。
10. **三段式 DataCopyPad 输出**：prologue/integral/epilogue 处理行循环跨越 head 边界的非对齐情况，同时完成 UB 连续→GM strided 布局转换。
11. **6 个 EVENT_ID 精细同步**：MTE2/MTE3/V 管道各事件独立标记，实现 DMA-Compute-DMA 三级重叠。
12. **三槽 dm 隔离**：PRE_LAUNCH=2 深度流水线下 dm 三槽互不冲突，Cube/Vector 核间延迟容忍。
13. **LSE 可选输出**：通过 LSE_MODE 模板参数编译期分支，不输出 LSE 时零开销。

---

## 12. 注意事项

- **本文件为实际运行版本**：flash_api.cpp 中所有 kernel 实例化均使用 `IntermCalcPrec=float`，对应本文件；rescale_o_low_prec.hpp（half 版）仅预留未被实例化。
- **hmUbTensor 在 rescale 中未使用**：它由 online_softmax 写入（局部 row max），rescale_o 不需要读（rescale 只用 dm/gl/gm），但仍分配以保持 UB 偏移与 softmax 一致。
- **`proTokenIdxPre` 声明但未使用**：operator() 中声明了 `uint32_t proTokenIdxPre=0` 但后续未引用，为预留变量。
- **`qSRemian` 声明但未使用**：`uint32_t qSRemian = qSThisSubBlock` 未被使用，为预留。
- **`HALF_DM_UB_SIZE`/`HALF_LL_UB_SIZE`/`MULTIPLIER`/`NUM4` 等常量预留**：编译期定义但当前 float 版本未使用，保留以兼容 half 版接口。
- **双视图地址安全**：Cast 后 half/bf16 数据只占 go32 前 16KB（float 占 32KB），DataCopyPad 仅读前半部分，不会越界。
- **lse32 复用 gl 时机**：lse32 在末块 Div 完成后才写入 gl 位置（gl 已被 Brcb+Div 消费），时序安全。
- **bf16 vs half 输出选择**：ElementOutput 由 OutputType_ 模板参数决定，flash_api.cpp 中根据用户 dtype 实例化为 bfloat16_t 或 half，影响 Cast 的 RoundMode。
- **高精度优势**：float 精度下 O 累加数值误差远小于 half 版本，尤其长序列场景下 dm*go 的精度损失可控，这也是默认使用 float 版本的原因。

---

## 13. 总结

`rescale_o.hpp` 以约 465 行代码实现了 FlashAttention 前向推理最后阶段的 Vector 核 epilogue（高精度 float 版本，实际使用版本），核心职责是：

1. **跨 KV stack 的 O 在线累加**：通过 `O = O*dm + OTmp` 公式在 float 精度下维护 O 分子累加器；
2. **末块归一化**：除以全局 rowsum l 得到最终 softmax 输出 O；
3. **float→half/bf16 类型转换**：bf16 用 CAST_RINT、half 用 CAST_NONE，原地写入双视图 go16；
4. **可选 LSE 输出**：`ln(l)+m` 全 float 精度直接计算，写回 GM；
5. **Sub-core 并行**：Decode 沿行 / Prefill 沿 head 的两维度拆分；
6. **UB 溢出处理**：行循环 + gUpdate GM 中转支持大行数场景；
7. **三段式输出**：prologue/integral/epilogue 处理 head 边界。

它与 `online_softmax.hpp`（float 版）通过共享 UB 偏移实现零拷贝数据传递，与 `pv_matmul.hpp` 通过 `CrossCoreWaitFlag(pvReady)` 跨核同步，共同构成 FlashAttention 前向推理的完整 Vector 核计算链路。作为实际实例化版本，本文件是理解 NPU FlashAttention 数值精度和性能特性的关键。
