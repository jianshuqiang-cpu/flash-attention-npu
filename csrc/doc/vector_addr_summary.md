# vector_addr.h 简要解析

## 文件职责

`csrc/flash_attn_npu/fag_common/vector_addr.h` 定义了 `VectorAddr<maskType, inputLayout>`，用于 FlashAttention 反向传播 FAG kernel 中 Vector 侧任务的块信息生成。

它不直接执行向量计算，而是生成 `VecAddrInfo` / `VecBlockInfo`，告诉 `EpilogueFAGOp` 当前要处理哪些 `128 x 128` attention 子块，以及这些子块对应的 batch、head、Q block、K block、workspace 偏移和尾块真实长度。

## 调用链路

```text
mha_varlen_bwd.cpp
  └─ VectorAddr::init()
  └─ VectorAddr::addr_mapping()
       └─ 生成 VecAddrInfo
            └─ EpilogueFAGOp::operator()(VecAddrInfo)
                 └─ Vector 侧处理 Cube workspace 中的中间结果
```

`VectorAddr` 与 `CubeAddr` 的遍历顺序需要保持一致：Cube 侧先把矩阵乘结果写入 workspace，Vector 侧再按相同的 blockId 读取并做 epilogue 处理。

## 核心数据结构

`VecAddrInfo` 定义在 `common_header.h` 中：

```cpp
struct VecAddrInfo {
    int32_t taskId;
    int32_t blockLength = 0;
    VecBlockInfo VecBlkInfo[16];
};
```

每个 `VecBlockInfo` 记录一个 Vector 子块：

```text
batchIdx    当前 batch
nheadsIdx   当前 query head
SeqQIdx     Q 方向 block 编号
SeqKIdx     K 方向 block 编号
nheadsKIdx  GQA/MQA 下对应的 KV head
gIdx        GQA/MQA 组内 query head
offset      当前 block 在 workspace 中的偏移
lengthy     Q 方向真实长度
lengthx     K 方向真实长度
```

## 核心流程

### 1. 初始化

`init()` 会设置：

- `batch`
- `nheads`
- `g`
- `headdim`
- `coreId`
- `qSeqlen / kSeqlen`
- `s1BlockNum / s2BlockNum`
- `s1TailLength / s2TailLength`
- `qSeqIdx / seqKIdx / batchIdx / nheadsIdx`

其中 Q/K 序列按 `128 token` 为基本粒度切块：

```text
s1BlockNum = ceil(qSeqlen / 128)
s2BlockNum = ceil(kSeqlen / 128)
```

### 2. 生成块信息

`addr_mapping()` 会遍历：

```text
batch -> head -> Q block -> K block
```

并在当前 segment 属于本 core 时写入 `VecAddrInfo`：

```cpp
if (coreSegmentBlockNum % coreNum == coreId) {
    getOffset(...);
}
```

这表示 segment 会按 core 轮转分配：

```text
segment0 -> core0
segment1 -> core1
segment2 -> core2
segment3 -> core0
```

### 3. causal 下三角过滤

VectorAddr 通过下面条件只保留 causal attention 的有效区域：

```cpp
if (qSeqIdx + y >= seqKIdx + x) {
    getOffset(...);
}
```

图示：

```text
         K0   K1   K2   K3
Q0       ✓    ×    ×    ×
Q1       ✓    ✓    ×    ×
Q2       ✓    ✓    ✓    ×
Q3       ✓    ✓    ✓    ✓
```

### 4. workspace 偏移

每个子块按 `128 x 128` 预留 workspace：

```cpp
vecPhyAddr.offset = blockId * 128 * 128;
```

因此 Vector 侧必须和 Cube 侧使用相同的 blockId 顺序，否则会读错 workspace。

## 示例一：256 长度的 causal attention

假设：

```text
qSeqlen = 256
kSeqlen = 256
s1BlockNum = 2
s2BlockNum = 2
s1GuardInterval = 2
qSeqIdx = 0
seqKIdx = 0
```

Q/K 都有两个 block：

```text
Q block: Q0, Q1
K block: K0, K1
```

causal 有效区域：

```text
         K0        K1
      +--------+--------+
Q0    | slot0  | skip   |
      +--------+--------+
Q1    | slot1  | slot2  |
      +--------+--------+
```

生成的 Vector 块信息大致是：

```text
slot0: SeqQIdx = 0, SeqKIdx = 0, offset = 0 * 128 * 128
slot1: SeqQIdx = 1, SeqKIdx = 0, offset = 1 * 128 * 128
slot2: SeqQIdx = 1, SeqKIdx = 1, offset = 2 * 128 * 128
```

## 示例二：300 长度的尾块处理

假设：

```text
qSeqlen = 300
kSeqlen = 300
```

则：

```text
Q0: 128 tokens
Q1: 128 tokens
Q2: 44 tokens

K0: 128 tokens
K1: 128 tokens
K2: 44 tokens
```

图示：

```text
         K0[128]   K1[128]   K2[44]
Q0[128]    ✓         ×         ×
Q1[128]    ✓         ✓         ×
Q2[44]     ✓         ✓         ✓
```

当处理尾块 `(Q2, K2)` 时，`getOffset()` 会生成：

```text
SeqQIdx = 2
SeqKIdx = 2
lengthy = 44
lengthx = 44
```

这样 Vector 侧只处理真实的 `44 x 44` 区域，避免越界。

## TND 与 BSND

`VectorAddr` 支持两种输入布局：

```cpp
InputLayout::BSND
InputLayout::TND
```

BSND 是定长 batch 布局，直接使用传入的 `seq_q_len / seq_k_len`。

TND 是变长序列布局，所有 token 在 T 维打平，需要通过 `cu_seqlens` 计算每个 batch 的真实长度：

```text
第 0 个样本长度 = cu_seq_qlen_addr[0]
第 i 个样本长度 = cu_seq_qlen_addr[i] - cu_seq_qlen_addr[i - 1]
```

调用方在 TND 模式下传入的是原始 `cu_seqlens + 1`，所以 `getSeqLen(0)` 读到的是第一个样本长度。

## 注意点

- `getLeftAddr()`、`getRightAddr()` 当前没有被 `addr_mapping()` 直接调用，更像是和 `CubeAddr` 对齐的统一接口或预留扩展。
- `getSeqRealLength()` 当前也没有被核心映射流程直接调用，尾块长度主要由 `getOffset()` 中的 `lengthx / lengthy` 处理。
- `blockLength` 和 `blockId` 的编号必须与 Cube 侧 workspace slot 保持一致。
- `VectorAddr` 每次最多描述 16 个 `128 x 128` 子块。
