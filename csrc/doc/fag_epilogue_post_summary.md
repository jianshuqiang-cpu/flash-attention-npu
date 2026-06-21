# fag_epilogue_post.hpp 简要解析

## 文件职责

`csrc/flash_attn_npu/fag_epilogue_post.hpp` 定义了 FlashAttention 反向传播 FAG 的最后一个 Vector 后处理模块：

```cpp
BlockEpilogue<
    EpilogueAtlasA2FAGPost,
    ElementVecDtype>
```

它在前面的 Cube/Vector 计算全部完成后执行，负责把 fp32 workspace 中累积好的 `dq/dk/dv` 转成最终输出 tensor。

核心职责：

```text
1. 从 fp32 workspace 读取 dq/dk/dv
2. 对 dq 和 dk 乘以 softmax_scale
3. 将 fp32 cast 到输出 dtype，例如 FP16/BF16
4. 写回最终 dq/dk/dv tensor
```

其中：

- `dq`：需要乘 `softmax_scale`
- `dk`：需要乘 `softmax_scale`
- `dv`：不需要乘 `softmax_scale`

## 在整体反向流程中的位置

在 `mha_varlen_bwd.cpp` 中，Post 阶段位于 Vector Op 和 Cube 计算之后：

```text
FAGPre
  ↓
FAGSfmg
  ↓
Cube/Vector 主循环
  ↓
等待 CUBE2POST
  ↓
FAGPost
  ↓
最终 dq/dk/dv 输出
```

对应调用链：

```text
AscendC::WaitEvent(CUBE2POST)
AscendC::SyncAll()
EpilogueFAGPost epilogueFagPost(...)
epilogueFagPost()
```

## 输入输出

### 输入 workspace

```cpp
AscendC::GlobalTensor<float> dqWorkSpaceGm;
AscendC::GlobalTensor<float> dkWorkSpaceGm;
AscendC::GlobalTensor<float> dvWorkSpaceGm;
```

这些 workspace 是前序阶段累积出的 fp32 梯度。

### 输出 tensor

```cpp
AscendC::GlobalTensor<ElementVecDtype> dqGm;
AscendC::GlobalTensor<ElementVecDtype> dkGm;
AscendC::GlobalTensor<ElementVecDtype> dvGm;
```

这些是最终写回给 PyTorch 的 `dq/dk/dv`。

图示：

```text
fp32 workspace                  final output tensor

 dqWorkSpaceGm  -- scale+cast -->  dqGm
 dkWorkSpaceGm  -- scale+cast -->  dkGm
 dvWorkSpaceGm  ------ cast ---->  dvGm
```

## 构造函数做了什么

构造函数主要完成四件事：

### 1. 读取 tiling 数据

从 `tiling_data` 中读取：

```text
dq workspace offset
dk workspace offset
dv workspace offset
qSize
kvSize
coreNum
softmax_scale
```

其中：

- `qSize`：dq 的展平元素数
- `kvSize`：dk/dv 的展平元素数

### 2. 绑定 GM tensor

```text
dq/dk/dv 输出 tensor
fp32 dq/dk/dv workspace
```

### 3. 计算 UB 单次处理容量

```cpp
ubBaseSize = ArchTag::UB_SIZE / POST_COEX_NODE / POST_BUFFER_NUM;
ubBaseSize = ubBaseSize / 256 * 256;
```

含义：

```text
把 UB 按 post 阶段的并存需求切开，并按 256 对齐。
```

### 4. 计算每个 core 负责的范围

dq 使用：

```text
qPostBlockTotal = qSize
qPostBaseNum = ubBaseSize / sizeof(float)
qPostBlockFactor = ceil(ceil(qSize / qPostBaseNum) / coreNum)
```

dk/dv 使用：

```text
kvPostBlockTotal = kvSize
kvPostBaseNum = qPostBaseNum
kvPostBlockFactor = ceil(ceil(kvSize / kvPostBaseNum) / coreNum)
```

## operator() 主流程

主入口：

```cpp
void operator()()
```

它分三段处理：

```text
1. 处理 dq
2. 处理 dk
3. 处理 dv
```

总体流程图：

```text
operator()
  │
  ├── 计算当前 core 的 dq 范围
  │     └── dq workspace -> scale -> cast -> dq output
  │
  ├── 计算当前 core 的 dk/dv 范围
  │
  ├── dk workspace -> scale -> cast -> dk output
  │
  └── dv workspace --------> cast -> dv output
```

## dq 后处理

dq 的处理逻辑：

```text
for i in 当前 core 负责的 dq 范围:
    1. 从 dqWorkSpaceGm 读取 fp32 数据到 UB
    2. vecIn *= scaleValue
    3. Cast fp32 -> ElementVecDtype
    4. 写回 dqGm
```

图：

```text
dqWorkSpaceGm(fp32)
        │
        ▼
      DataCopy
        │
        ▼
      UB fp32
        │
        ├── * softmax_scale
        │
        ├── Cast to FP16/BF16
        │
        ▼
      dqGm
```

## dk 后处理

dk 与 dq 类似，也需要乘 `softmax_scale`：

```text
dkWorkSpaceGm(fp32)
        │
        ▼
      DataCopy
        │
        ▼
      UB fp32
        │
        ├── * softmax_scale
        │
        ├── Cast to FP16/BF16
        │
        ▼
      dkGm
```

## dv 后处理

dv 不需要乘 `softmax_scale`，只需要 cast 和写回：

```text
dvWorkSpaceGm(fp32)
        │
        ▼
      DataCopy
        │
        ▼
      UB fp32
        │
        ├── Cast to FP16/BF16
        │
        ▼
      dvGm
```

原因可以理解为：

```text
dq/dk 来自 score 方向梯度，需要补 softmax_scale；
dv 来自 P^T @ dO，不经过 score 缩放路径，因此不乘 scale。
```

## 示例一：qSize 很小，单 core 一次处理完

假设：

```text
qSize = 1024
qPostBaseNum = 4096
coreNum = 8
```

则：

```text
qPostBlockOuterTotal = ceil(1024 / 4096) = 1
qPostBlockFactor = ceil(1 / 8) = 1
```

core 0：

```text
qBegin = 0 * 1 * 4096 = 0
qEnd = min(4096, 1024) = 1024
```

core 1 及之后：

```text
qBegin >= 4096 > qSize
循环不执行
```

图：

```text
qSize = 1024

[ core0: 0 ~ 1023 ]
[ core1: empty    ]
[ core2: empty    ]
...
```

## 示例二：qSize 较大，多个 core 分段处理

假设：

```text
qSize = 20000
qPostBaseNum = 4096
coreNum = 4
```

先计算 outer block 数：

```text
qPostBlockOuterTotal = ceil(20000 / 4096) = 5
qPostBlockFactor = ceil(5 / 4) = 2
```

所以每个 core 最多处理 2 个 outer block：

```text
core0: block 0~1 -> 元素 0 ~ 8191
core1: block 2~3 -> 元素 8192 ~ 16383
core2: block 4   -> 元素 16384 ~ 19999
core3: empty
```

图：

```text
qSize = 20000, qPostBaseNum = 4096

0                                                   19999
|-------------------------------------------------------|
| core0      | core0      | core1      | core1      |core2|
| 0~4095     |4096~8191   |8192~12287  |12288~16383 |tail |
```

最后一个 tail block：

```text
dataSize = qPostTailNum = 20000 % 4096 = 3616
```

## 对齐规则

### fp32 输入对齐

```cpp
DataCopy(vecIn, workspace[i], (dataSize + 7) / 8 * 8);
```

fp32 一个元素 4 字节，8 个元素是 32B，因此 fp32 输入按 8 元素对齐。

### FP16/BF16 输出对齐

```cpp
DataCopy(output[i], vecOut, (dataSize + 15) / 16 * 16);
```

FP16/BF16 一个元素 2 字节，16 个元素是 32B，因此输出按 16 元素对齐。

## 与 FAGPre 的关系

`FAGPre` 在反向开始时清零 workspace：

```text
FAGPre:
    dqWorkSpace = 0
    dkWorkSpace = 0
    dvWorkSpace = 0
```

中间 Cube/Vector 阶段向 workspace 累积梯度：

```text
Cube/Vector:
    accumulate dq/dk/dv in fp32 workspace
```

最后 `FAGPost` 输出最终梯度：

```text
FAGPost:
    fp32 workspace -> scale/cast -> final dq/dk/dv
```

图：

```text
FAGPre 清零
    ↓
Cube/Vector 累积 fp32 梯度
    ↓
FAGPost 缩放、cast、写回
```

## 注意点

- `dq` 和 `dk` 会乘以 `softmax_scale`。
- `dv` 不乘 `softmax_scale`。
- workspace 内部使用 fp32 累积，最终输出转成 `ElementVecDtype`。
- `qSize` 用于 dq，`kvSize` 用于 dk/dv。
- 每个 core 通过 `cBlockIdx` 计算自己负责的展平元素范围。
- fp32 输入按 8 元素对齐，FP16/BF16 输出按 16 元素对齐。
