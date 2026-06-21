# fag_epilogue_op.hpp 简要解析

## 文件职责

`csrc/flash_attn_npu/fag_epilogue_op.hpp` 定义了 FlashAttention 反向传播 FAG 中 Vector 侧的核心 epilogue：

```cpp
BlockEpilogue<
    EpilogueAtlasA2FAGOp,
    ElementVecDtype,
    std::integral_constant<InputLayout, inputLayout>>
```

它消费 `VectorAddr` 生成的 `VecAddrInfo`，从 Cube 侧 workspace 读取中间结果，完成两个关键任务：

1. 重新计算 softmax 概率 `P`
2. 计算 softmax 反向中的 `dS`

简化公式：

```text
SubGrapA:
    scores -> scale -> mask -> softmax -> P

SubGrapB:
    dP -> dP - rowsum(dP * P) -> * P -> dS
```

其中：

- `P` 写入 `dropWorkSpaceGm`
- `dS` 写入 `mulWorkSpaceGm`

后续 Cube2 / Cube3 会继续使用这些 workspace 来计算 `dq / dk / dv`。

## 整体调用链

`fag_epilogue_op.hpp` 在 `mha_varlen_bwd.cpp` 的 Vector 分支中被调用：

```text
VectorAddr::addr_mapping()
        ↓
生成 VecAddrInfo / VecBlockInfo
        ↓
EpilogueFAGOp::operator()(VecAddrInfo)
        ↓
SubGrapA: 重算 P
SubGrapB: 计算 dS
        ↓
通知 Cube 侧继续计算 dq/dk/dv
```

## 输入输出 workspace

主要 GM tensor：

```cpp
GlobalTensor<float> mm1WorkspaceGm;
GlobalTensor<float> mm2WorkspaceGm;
GlobalTensor<ElementVecDtype> dropWorkSpaceGm, mulWorkSpaceGm;
GlobalTensor<float> rowLseGm;
GlobalTensor<float> sfmgWorkspaceGm;
```

含义：

```text
mm2WorkspaceGm
    Cube 写入的 score 类中间结果
    ↓
    SubGrapA 读取
    ↓
dropWorkSpaceGm
    Vector 写入的 softmax 概率 P

mm1WorkspaceGm
    Cube 写入的 dP 类中间结果
    ↓
    SubGrapB 读取
    ↓
mulWorkSpaceGm
    Vector 写入的 dS
```

图示：

```text
Cube1
  ├── mm2Workspace: score block
  │       ↓
  │    SubGrapA
  │       ↓
  │    dropWorkSpace: P
  │
  └── mm1Workspace: dP-like block
          ↓
       SubGrapB
          ↓
       mulWorkSpace: dS
```

## Vector core 与 Cube block 的对应关系

构造函数中：

```cpp
blockIdx = GetBlockIdx();
cubeBlockIdx = blockIdx / 2;
subIdx = blockIdx % 2;
```

含义是：两个 Vector core 对应一个 Cube 逻辑 core。

一个 `128 x 128` Cube block 会沿 Q 方向切成上下两半：

```text
一个 128 x 128 attention 子块

+--------------------------+
| subIdx = 0               | 处理上半部分 Q 行
| 大约 64 x 128             |
+--------------------------+
| subIdx = 1               | 处理下半部分 Q 行
| 大约 64 x 128             |
+--------------------------+
```

对应代码：

```cpp
s1VecSize = (s1CubeExtend + 1) / 2;
s1Extend = subIdx ? s1CubeExtend - s1VecSize : s1VecSize;
```

完整块示例：

```text
s1CubeExtend = 128

subIdx = 0 -> s1Extend = 64
subIdx = 1 -> s1Extend = 64
```

尾块示例：

```text
s1CubeExtend = 45
s1VecSize = (45 + 1) / 2 = 23

subIdx = 0 -> s1Extend = 23
subIdx = 1 -> s1Extend = 22
```

## operator() 主流程

主入口：

```cpp
void operator()(const VecAddrInfo &addrs)
```

它的流程：

```text
1. 读取 taskId，计算 pingpongIdx
2. 如果是 task 0，预取 causal mask 到 UB
3. 遍历当前 VecAddrInfo 中的每个 VecBlockInfo
4. 根据 blockInfo 计算当前子块的形状和偏移
5. 调用 SubGrapA 重新计算 P
6. 调用 SubGrapB 计算 dS
```

流程图：

```text
operator()(VecAddrInfo)
    │
    ├── taskId -> pingpongIdx
    ├── 首个 task 预取 causal mask
    │
    └── for each VecBlockInfo
          │
          ├── 计算 s1Extend / s2Extend / s2ExtendAlign
          ├── 计算 lseOffset
          ├── 计算 sfmgOffset
          ├── 计算 copyInOffset / copyOutOffset
          │
          ├── SubGrapA
          │     └── score -> scale -> mask -> softmax -> P
          │
          └── SubGrapB
                └── dP -> dP - sfmg -> * P -> dS
```

## SubGrapA：重新计算 softmax 概率 P

`SubGrapA` 的流程：

```text
1. 从 rowLseGm 读取 LSE / softmax 辅助信息
2. 从 mm2WorkspaceGm 读取 score block
3. score *= softmax_scale
4. 如果当前块在 causal 对角线上，应用 causal mask
5. 计算 softmax，得到 P
6. cast 到 ElementVecDtype
7. 写入 dropWorkSpaceGm
```

图：

```text
mm2WorkspaceGm
      │
      ▼
score block
      │
      ├── * softmax_scale
      │
      ├── causal mask，只对 SeqQIdx == SeqKIdx 的对角块
      │
      ▼
softmax
      │
      ▼
P
      │
      ▼
dropWorkSpaceGm
```

只对对角块做 mask 的原因：

```text
严格下三角块：全部有效
对角线块：块内部有一半需要 mask
上三角块：VectorAddr 不会生成
```

图示：

```text
         K0        K1        K2
Q0       diag      skip      skip
Q1       full      diag      skip
Q2       full      full      diag
```

## SubGrapB：计算 softmax 反向 dS

`SubGrapB` 的流程：

```text
1. 从 sfmgWorkspaceGm 读取 softmax gradient 辅助项
2. 从 mm1WorkspaceGm 读取 dP 类中间结果
3. 做逐行广播相减：
       dP - rowsum(dP * P)
4. 乘以上一步 SubGrapA 得到的 P：
       dS = P * (dP - rowsum(dP * P))
5. cast 到 ElementVecDtype
6. 写入 mulWorkSpaceGm
```

图：

```text
mm1WorkspaceGm                  sfmgWorkspaceGm
      │                                │
      ▼                                ▼
     dP                    rowsum(dP * P)
      │                                │
      └────────── sub ────────────────┘
                    │
                    ▼
          dP - rowsum(dP * P)
                    │
                    │   P
                    ▼
        P * (dP - rowsum(dP * P))
                    │
                    ▼
                  dS
                    │
                    ▼
              mulWorkSpaceGm
```

## 示例一：完整 128x128 块

假设当前 `VecBlockInfo` 是完整块：

```text
lengthy = 128
lengthx = 128
```

两个 Vector core 分摊：

```text
subIdx = 0:
    s1VecSize = 64
    s1Extend = 64
    s2Extend = 128

subIdx = 1:
    s1VecSize = 64
    s1Extend = 64
    s2Extend = 128
```

图：

```text
原始 Cube block: 128 x 128

              K 方向 128
        +--------------------+
Q 64    | subIdx = 0         |
        | 计算 P 和 dS       |
        +--------------------+
Q 64    | subIdx = 1         |
        | 计算 P 和 dS       |
        +--------------------+
```

两个 Vector core 读取同一个逻辑 block 的不同 Q 行区域：

```cpp
copyInOffset =
    cubeBlockIdx * cubeBaseMN * 2
  + pingpongIdx * cubeBaseMN
  + blockInfo.offset
  + curSeqQIdx * s1VecSize * s2CubeExtend;
```

其中 `curSeqQIdx = subIdx`。

## 示例二：尾块 45x100

假设当前 `VecBlockInfo` 表示一个尾块：

```text
lengthy = 45
lengthx = 100
```

那么：

```text
s1CubeExtend = 45
s2Extend = 100
s2ExtendAlign = align16(100) = 112
```

两个 Vector core 分摊：

```text
s1VecSize = (45 + 1) / 2 = 23

subIdx = 0:
    s1Extend = 23
    处理 23 x 100，有效列对齐到 112

subIdx = 1:
    s1Extend = 22
    处理 22 x 100，有效列对齐到 112
```

图：

```text
尾块: 45 x 100

              K 方向 100，有效计算
              K 方向 112，对齐搬运
        +------------------------+
Q 23    | subIdx = 0             |
        +------------------------+
Q 22    | subIdx = 1             |
        +------------------------+
```

`K` 方向需要 16 对齐：

```cpp
s2ExtendAlign = (s2Extend + 15) / 16 * 16;
```

因此：

```text
真实有效 K 长度: 100
对齐搬运/计算长度: 112
```

## TND / BSND 布局差异

`operator()` 中会根据 `inputLayout` 计算当前 batch 的真实长度：

```cpp
if constexpr (getLayout() == InputLayout::TND) {
    GetSeqQlenKvlenByBidx(blockInfo.batchIdx, cuQSeqLen, cuKSeqLen);
} else {
    cuQSeqLen = seq_q;
    cuKSeqLen = seq_k;
}
```

### BSND

定长 batch，序列长度统一：

```text
cuQSeqLen = seq_q
cuKSeqLen = seq_k
```

### TND

变长 batch，通过 `cu_seqlens` 计算每个 batch 的真实长度：

```text
batch 0 长度 = cu_seq[0]
batch i 长度 = cu_seq[i] - cu_seq[i - 1]
```

这会影响：

- `lseOffset`
- `sfmgOffset`
- 当前 block 的真实 Q/K 范围

## 与 vector_addr.h 的关系

`vector_addr.h` 负责生成：

```cpp
VecAddrInfo
```

其中每个 `VecBlockInfo` 包含：

```text
batchIdx
nheadsIdx
SeqQIdx
SeqKIdx
nheadsKIdx
gIdx
offset
lengthy
lengthx
```

`fag_epilogue_op.hpp` 使用这些信息来计算：

```text
当前子块在哪里？
当前子块多大？
当前子块属于哪个 batch/head？
从 workspace 哪里读？
写回 workspace 哪里？
是否需要 causal mask？
```

关系图：

```text
VectorAddr
   │
   ├── 决定要处理哪些 block
   ├── 决定 block 在 workspace 中的 offset
   └── 决定 block 的真实 lengthx / lengthy
        ↓
FAGOp
   │
   ├── 根据 offset 读 Cube workspace
   ├── 根据 lengthx / lengthy 处理尾块
   ├── 根据 SeqQIdx / SeqKIdx 判断是否要 mask
   └── 写回 P 和 dS
```

## 注意点

- `SubGrapA` 负责重新计算 `P`。
- `SubGrapB` 负责根据 `P` 和 `sfmg` 计算 `dS`。
- `SeqQIdx == SeqKIdx` 的对角块需要 causal mask。
- 严格下三角块全部有效，不需要 mask。
- 上三角块不会由 `VectorAddr` 生成。
- `taskId % 2` 用于 ping-pong workspace。
- 一个 `128 x 128` Cube block 会被两个 Vector core 沿 Q 方向分半处理。
