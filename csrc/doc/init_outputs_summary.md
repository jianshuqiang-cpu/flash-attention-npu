# init_outputs.hpp 详细解析

## 1. 文件定位

[init_outputs.hpp](../flash_attn_npu/init_outputs.hpp) 是 FlashAttention-NPU **前向推理**中负责**输出张量初始化**的 Epilogue 组件。它是 CATLASS 框架 `BlockEpilogue` 类模板的一个特化，策略标签为 `EpilogueAtlasA2InitOutWhenZero`（定义在 [fa_block.h](../flash_attn_npu/fa_block.h)）。

```text
┌─────────────────────────────────────────────────────────────┐
│            FlashAttention 前向 Vector 核 Epilogue            │
├─────────────────────────────────────────────────────────────┤
│  EpilogueInitOutWhenZero  ← init_outputs.hpp（本文件）       │
│       ↓ 初始化 O=0, LSE=+inf                                 │
│  EpilogueOnlineSoftmax   ← online_softmax.hpp               │
│       ↓ scale / mask / exp / rowmax / rowsum                 │
│  EpilogueRescaleO        ← rescale_o.hpp                    │
│       ↓ O 重缩放 + 累加 + 归一化                              │
└─────────────────────────────────────────────────────────────┘
```

### 在前向 Pipeline 中的位置

```text
Host 侧 (flash_api.cpp)
    │
    │ 构造 FAInferTilingData，分配 workspace
    ▼
Kernel 入口 (mha_fwd_kvcache.cpp, Vector 核)
    │
    ├── [若 kvSLoopNumTotal <= 0]
    │     epilogueInitOut()  ← 本文件：直接将 O 清零、LSE 置 inf
    │
    └── [KV 分块循环]
          ├── Cube: Q×K^T → S
          ├── Vector: epilogueOnlineSoftmax(S → P)
          ├── Cube: P×V → OTmp
          └── Vector: epilogueRescaleO(OTmp → O 累加/归一化)
```

---

## 2. 为什么需要初始化？

FlashAttention 使用**在线 softmax**（online softmax）算法在 KV 分块上流式计算注意力：

| 阶段 | 操作 | 对 O 的影响 |
|------|------|------------|
| 首个 KV 分块 | `O = P × V` | 直接赋值 |
| 中间分块 | `O = O × exp(dm) + P × V` | 需要读取旧 O 做重缩放 |
| 最后分块 | `O = O / rowsum` | 归一化 |
| LSE | `LSE = rowmax + log(rowsum)` | 需要正确的初始 rowmax |

因此：
- **O 必须初始化为 0**：否则全局内存（GM）中的垃圾值会在第一个分块被累加到结果中。
- **LSE 必须初始化为 +∞（或极大值）**：标记"尚未处理任何 KV 分块"，使得第一个分块的 `rowmax` 能被正确设置为该分块 S 的最大值，而不是与垃圾值比较。

> **注意**：对于正常有 KV 数据的路径，首个分块的初始化实际上由 `rescale_o.hpp` 中"首块直接赋值"（`go = lo`）隐式完成。本 Epilogue 主要处理 **kvSLoopNumTotal <= 0**（无有效 KV 序列）的边界情况。

---

## 3. 类结构与模板参数

```cpp
template <class AttnOutType_, class LseOutType_, LseModeT LSE_MODE_>
class BlockEpilogue<EpilogueAtlasA2InitOutWhenZero<LSE_MODE_>, AttnOutType_, LseOutType_>
```

| 模板参数 | 含义 | 典型值 |
|----------|------|--------|
| `AttnOutType_` | 输出 O 的类型（元素+布局） | `GemmType<half, LayoutBSND>` 或 `GemmType<bfloat16_t, ...>` |
| `LseOutType_` | LSE 输出的类型（元素+布局） | `GemmType<float, LayoutLse>` |
| `LSE_MODE_` | LSE 输出模式 | `OUT_ONLY`（输出 LSE）或 `NONE`（不输出 LSE） |

### 关键常量

| 常量 | 值 | 含义 |
|------|----|------|
| `ATTN_OUT_INI` | `0.0f` | 注意力输出 O 的初始值 |
| `LSE_OUT_INI` | `3e+99f` | LSE 初始值（接近 float 最大值，表示"未处理"） |
| `UB_UINT8_BLOCK_SIZE` | `16384` | UB 逻辑块大小（16KB） |
| `ATTN_OUT_INIT_UB_TENSOR_OFFSET` | `0` | O 在 UB 中的起始偏移 |
| `LSE_OUT_INIT_UB_TENSOR_OFFSET` | `6 * 16384 = 98304` | LSE 在 UB 中的起始偏移 |

---

## 4. 核心流程详解

### 4.1 构造函数：UB 空间分配

```cpp
BlockEpilogue(Arch::Resource<ArchTag> &resource)
{
    attnOutUbTensor = resource.ubBuf.GetBufferByByte<ElementAttnOut>(0);
    lseOutUbTensor = resource.ubBuf.GetBufferByByte<ElementLseOut>(6 * UB_UINT8_BLOCK_SIZE);
}
```

三个 Vector 核 Epilogue（InitOut、OnlineSoftmax、RescaleO）共享同一个 UB，通过不同偏移分配各自的临时缓冲区：

```text
UB 布局（按偏移，单位：字节）
┌────────────────────────────────────────────────────────────┐
│ 0 ~ 98303     │ InitOut.attnOutUbTensor / OnlineSoftmax 共享区域  │
│ 98304 ~       │ InitOut.lseOutUbTensor / RescaleO 部分区域         │
│ ...           │ （其他 epilogue 张量）                              │
└────────────────────────────────────────────────────────────┘
```

### 4.2 operator()：Sub-core 任务拆分

`operator()` 是 Epilogue 的入口，负责将一个 query block（`qSBlockSize × qNBlockSize`）拆分给多个 sub-core 并行执行。

**拆分策略**：

```text
情况 A：qNBlockSize == 1（head 维只有 1 个 tile）→ 按 S 维（序列行）拆分
─────────────────────────────────────────────────────────
  qSBlockSize 行
  ┌─────┐
  │sub-0│ ← rowSplitSubBlock = qSBlockSize / subBlockNum 行
  ├─────┤
  │sub-1│ ← rowNum - rowSplitSubBlock 行（余数）
  └─────┘
  列偏移 = 0，行偏移递增

情况 B：qNBlockSize > 1（head 维有多个 tile）→ 按 N 维（head）拆分
─────────────────────────────────────────────────────────
  qSBlockSize 行 × qNSplitSubBlock heads
  ┌─────┬─────┐
  │sub-0│sub-1│  每个 sub-core 负责 qNSplitSubBlock 个 head
  └─────┴─────┘
   所有行  列偏移 = subBlockIdx * qNSplitSubBlock * embedV
```

拆分后，每个 sub-core 计算自己负责的 GM 偏移，调用 `SubCoreCompute` 执行实际初始化。

### 4.3 SubCoreCompute：初始化核心逻辑

`SubCoreCompute` 执行两个独立的初始化步骤，使用不同的硬件事件 ID 做流水线同步：

```text
                    EVENT_ID6                    EVENT_ID7
                       │                            │
  ┌────────────────────▼──────────────────┐  ┌──────▼──────────────────────┐
  │  初始化 O（注意力输出）为 0             │  │  初始化 LSE 为 +inf          │
  │                                        │  │  （仅 LSE_MODE=OUT_ONLY）   │
  │  1. WaitFlag(MTE3→V, ID6)             │  │  1. WaitFlag(MTE3→V, ID7)  │
  │  2. Duplicate(ub, 0, N)               │  │  2. Duplicate(ub, +inf, N) │
  │  3. SetFlag(V→MTE3, ID6)             │  │  3. SetFlag(V→MTE3, ID7)   │
  │  4. WaitFlag(V→MTE3, ID6)            │  │  4. WaitFlag(V→MTE3, ID7)  │
  │  5. DataCopyPad(ub → gOutput) × heads │  │  5. DataCopyPad(ub→gLse)   │
  │  6. SetFlag(MTE3→V, ID6)             │  │  6. SetFlag(MTE3→V, ID7)   │
  └────────────────────────────────────────┘  └─────────────────────────────┘
```

**WaitFlag / SetFlag 同步模式解释**：
- `WaitFlag<MTE3_V>(ID)`：等待 MTE3（DMA 写通道）完成对 GM 的写操作，Vector 管道才能安全读取/覆盖 UB
- `SetFlag<V_MTE3>(ID)`：通知 MTE3 通道，UB 数据已准备好，可以开始 DMA 搬运
- 这种乒乓式 flag 同步确保了 Vector 计算和 DMA 搬运之间的正确依赖关系

**DataCopyPad 参数解析（以 O 初始化为例）**：

```cpp
DataCopyPad(
    gOutput[qNIdx * embedV],                    // 目的地址：第 qNIdx 个 head 的起始位置
    attnOutUbTensor,                             // 源地址：UB 中已填充 0 的张量
    DataCopyExtParams(
        qSThisSubBlock,                          // 拷贝行数（query 序列维）
        embedV * sizeof(ElementAttnOut),         // 每行连续字节数 = head_dim × sizeof(half)
        0,                                        // 源行间隔（UB 中连续排列）
        (oHiddenSize - embedV) * sizeof(...),    // 目的行间隔 = (总heads-1) * head_dim（跳到下一个 token 的同 head）
        0                                         // 目的起始偏移
    )
);
```

对于 LSE，每行只有 1 个 float 元素，所以目的行间隔为 `(qHeads - 1) * sizeof(float)`。

---

## 5. 例子 1：标准 BSND 前向零 KV 边界情况

假设一个标准前向推理场景（无有效 KV cache）：

```text
Q 形状: [batch=1, qSeqlen=128, num_heads=8, head_dim=64]
K/V 形状: [batch=1, kvSeqlen=0, num_heads_k=8, head_dim=64]  ← kvSeqlen=0
```

此时 `kvSLoopNumTotal <= 0`，`epilogueInitOut` 被调用：

```text
qSBlockSize = 128, qNBlockSize = ?（由 GetQNBlockTile 决定，假设为 2）
subBlockNum = 2（Vector 核 sub-core 数）
```

**拆分过程（按 N 维拆分，因为 qNBlockSize=2 > 1）**：

```text
O 张量布局（BSND 展开为 [total_tokens, oHiddenSize]）:
  total_tokens = 128, oHiddenSize = 8 * 64 = 512
  embedV = 64

  ┌──────────┬──────────┬──────────┬────────┬──────────┐
  │ head_0   │ head_1   │ head_2   │  ...   │ head_7   │
  │ 64 elems │ 64 elems │ 64 elems │        │ 64 elems │
  └──────────┴──────────┴──────────┴────────┴──────────┘
  ←────────────── 512 half 元素 ──────────────────────→

sub-0 (subBlockIdx=0):
  qNThisSubBlock = qNSplitSubBlock = 2/2 = 1 个 head（head 0,1 中的第1组）
  qSThisSubBlock = 128 行
  列偏移 = 0 * 1 * 64 = 0 → 写 head 0~1 区域

sub-1 (subBlockIdx=1，余数):
  qNThisSubBlock = 2 - 1 = 1 个 head
  qSThisSubBlock = 128 行
  列偏移 = 1 * 1 * 64 = 64 → 写 head 2~3 区域...
  （实际 qNSplitSubBlock 和余数计算取决于 qNBlockSize/subBlockNum 的整除关系）
```

**执行结果**：
- O 的所有 128 × 512 个 half 元素被写为 0
- LSE 的所有 128 × 8 个 float 元素被写为 3e+99

---

## 6. 例子 2：GQA 场景单 head tile 按行拆分

假设 GQA 场景，`qNBlockSize = 1`（head 维只有 1 个 tile），按 S 维拆分：

```text
qSBlockSize = 128, qNBlockSize = 1
num_heads = 32, num_heads_k = 8（groupSize = 4）
head_dim = 128
subBlockNum = 2
```

**拆分过程（按 S 维拆分，因为 qNBlockSize=1）**：

```text
O 张量形状: [128, 32*128=4096]

  ┌─────────────────────────────────────┐  row 0
  │              head 0~31              │
  ├─────────────────────────────────────┤  row 63
  │           sub-0 (64 行)             │
  ├─────────────────────────────────────┤  row 64
  │           sub-1 (64 行)             │
  └─────────────────────────────────────┘  row 127

sub-0:
  rowSplitSubBlock = 128/2 = 64 行
  rowOffsetSubBlock = 0*64 = 0
  outRowOffset = 0, outColOffset = 0 → 从 O[0, 0] 开始写 64 行

sub-1 (subBlockIdx=1，余数):
  rowActualSubBlock = 128 - 64 = 64 行
  rowOffsetSubBlock = 1*64 = 64
  outRowOffset = 64, outColOffset = 0 → 从 O[64, 0] 开始写 64 行
```

**LSE 写入模式**：
LSE 形状为 `[128, 32]`（每个 head 一个 float），`DataCopyPad` 的行间隔为 `(32-1)*sizeof(float) = 124` 字节，所以每次写 1 个 float，跳过 124 字节到下一个 token 的同一个 head。

```text
LSE 布局（float32）:
        head_0   head_1  ...  head_31
  row 0 [+inf]   ...            ...
  row 1 [+inf]   ...
  ...
  row 63 [+inf]  ...            ...     ← sub-0 写前 64 行
  ---
  row 64 [+inf]  ...            ...
  ...
  row 127[+inf]  ...            ...     ← sub-1 写后 64 行
```

---

## 7. 数据流图

```text
                  ┌──────────────────┐
                  │  全局内存 (GM)    │
                  │  gOutput (O)     │ ← 初始为垃圾值
                  │  gLse            │ ← 初始为垃圾值
                  └────────┬─────────┘
                           │
    operator()             │
  ┌────────────────────────▼────────────────────────┐
  │  1. 解析 layout，获取 oHiddenSize/qHeads/embedV  │
  │  2. 获取 subBlockIdx / subBlockNum              │
  │  3. 按 qNBlockSize 选择拆分维度：                │
  │     - ==1: 按 S 维拆行                          │
  │     - >1 : 按 N 维拆列(head)                    │
  │  4. 计算当前 sub-core 的 GM 偏移                 │
  │  5. 调用 SubCoreCompute                         │
  └────────────────────────┬────────────────────────┘
                           │
                  SubCoreCompute
  ┌────────────────────────▼────────────────────────┐
  │  ┌─────────────────────────────────────┐        │
  │  │ 初始化 O:                           │        │
  │  │  Duplicate(ub, 0) → DataCopyPad → GM│        │
  │  │  EVENT_ID6 做 V↔MTE3 同步          │        │
  │  └─────────────────────────────────────┘        │
  │                    │                            │
  │  ┌─────────────────▼───────────────────┐        │
  │  │ [if LSE_MODE=OUT_ONLY]              │        │
  │  │ 初始化 LSE:                         │        │
  │  │  Duplicate(ub, 3e99) → DataCopyPad  │        │
  │  │  EVENT_ID7 做 V↔MTE3 同步          │        │
  │  └─────────────────────────────────────┘        │
  └────────────────────────┬────────────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  全局内存 (GM)    │
                  │  gOutput = 0     │ ← 已清零
                  │  gLse = +inf     │ ← 已置极大值
                  └──────────────────┘
```

---

## 8. UB 与 GM 数据搬运细节

### Duplicate（Vector 指令）
在 UB 上用标量值填充一段连续内存：
```cpp
Duplicate(tensor, value, count);
// tensor: LocalTensor（UB 上）
// value: 填充值（0 或 3e+99）
// count: 填充元素个数
```

### DataCopyPad（MTE3 DMA 指令）
带行跨度的 2D DMA 搬运，从 UB 写到 GM：
```cpp
DataCopyPad(dst, src, DataCopyExtParams(rows, srcStride, gapSrc, gapDst, dstOffset));
// rows    : 搬运行数
// srcStride: 源地址每行连续字节数
// gapSrc   : 源地址行间隔
// gapDst   : 目的地址行间隔（用于跨 head 跳跃）
// dstOffset: 目的地址起始偏移
```

对于 O 初始化（每 head 的 head_dim=64 个 half）：
- `srcStride = 64 * 2 = 128` 字节（每 head 每行 128 字节连续）
- `gapSrc = 0`（UB 中连续排列）
- `gapDst = (oHiddenSize - 64) * 2` 字节（跳到下一个 token 的同 head）

对于 LSE 初始化（每 head 1 个 float）：
- `srcStride = 4` 字节（1 个 float）
- `gapSrc = 0`
- `gapDst = (qHeads - 1) * 4` 字节（跳到下一个 token 的同 head）

---

## 9. 注意点

1. **仅 Vector 核执行**：该 Epilogue 在 `#ifdef __DAV_C220_VEC__` 保护下实例化，Cube 核不参与。
2. **边界情况处理**：主要用于 `kvSLoopNumTotal <= 0`（无有效 KV 序列）时，确保输出是确定的零值而非垃圾值。
3. **UB 偏移协调**：`attnOutUbTensor` 使用 UB 偏移 0，`lseOutUbTensor` 使用偏移 6×16KB，与 `online_softmax.hpp` 和 `rescale_o.hpp` 的 UB 使用区域协调，通过流水线阶段互斥访问避免冲突。
4. **LSE 初始值 3e+99**：不是严格的 +inf，但在 float32 范围内足够大（接近 float 最大值 3.4e+38），能确保第一个分块的 `rowmax` 比较正确。
5. **embedRoundV 对齐**：`embedV` 会按 `HALF_ELEM_NUM_PER_BLK=16` 向上对齐，确保 Duplicate/DataCopyPad 的块对齐要求。
6. **v2 vs v3**：v3 版本（`flash_attn_npu_v3`）移除了这个独立的 InitOut Epilogue，推测零 KV 边界情况通过其他方式处理。
7. **Sub-core 余数处理**：`subBlockIdx == 1` 的 core 负责余数（`rowNum - rowSplitSubBlock`），这是一种简化的负载分配策略，当 subBlockNum > 2 时其余 core 均分。

---

## 10. 总结

`init_outputs.hpp` 是 FlashAttention-NPU 前向推理流水线中的**第一个 Vector 核 Epilogue**，职责非常单一但必要：

- **功能**：将注意力输出 O 初始化为 0，LSE 初始化为 +inf
- **机制**：利用 AscendC 的 `Duplicate`（Vector 填充）+ `DataCopyPad`（MTE3 DMA 2D 搬运）实现 UB→GM 的初始化写入
- **并行**：通过 `operator()` 中的 sub-core 拆分逻辑，支持 Vector 核的多 sub-core 并行
- **同步**：使用 `WaitFlag/SetFlag` 硬件事件实现 Vector 计算与 DMA 搬运的流水线同步
- **定位**：属于边界情况处理组件，正常 KV 路径的初始化由 `rescale_o.hpp` 首块赋值隐式完成
