# fag_sfmg.h 解析

## 1. 文件定位

`fag_sfmg.h` 是 FAG backward 中 sfmg 预处理阶段使用的一个向量侧辅助头文件，核心对外接口是：

```cpp
template<typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxGradFront(
    const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &gradTensor,
    const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SoftMaxShapeInfo &softmaxShapeInfo = {})
```

它的主要作用是对每个 token/head 行计算：

```text
sfmg[row] = sum_j(gradTensor[row, j] * srcTensor[row, j])
```

在当前 FAG 反向链路里，调用方是 `fag_epilogue_sfmg.hpp`，其中：

```text
gradTensor ≈ dout
srcTensor  ≈ out
```

因此它实际计算的是每行的：

```text
sum(dout * out)
```

这个值后续会作为 softmax backward 的前置行归约项，参与 `dS` 的计算。

---

## 2. 在 FAG backward 中的位置

整体关系可以理解为：

```text
forward 输出 out / backward 输入 dout
        │
        ▼
FAGSfmg 阶段
        │
        ├── Cast dout -> float
        ├── Cast out  -> float
        └── SoftmaxGradFront(float)
                │
                ├── 逐元素乘：dout * out
                ├── 沿 headdim 归约：sum(dout * out)
                └── 每行广播成 8 个 float，写入 sfmg workspace
        │
        ▼
FAGOp / vector epilogue
        │
        └── 使用 sfmg workspace 计算 softmax 反向中的 dS
```

`sfmg workspace` 每行保存 8 个 `float`，也就是 32B：

```text
一行 token/head:

sum(dout * out) = s

写入 workspace:

[s, s, s, s, s, s, s, s]
```

这样做的目的通常是满足向量搬运和后续读取的 32B 对齐需求。

---

## 3. 核心数学含义

Softmax backward 中经常需要如下行归约项：

```text
row_sum = sum_j(dY_j * Y_j)
```

其中：

- `Y` 是 softmax 输出；
- `dY` 是传入 softmax 的梯度。

在 FlashAttention backward 的语境中，可以类比为：

```text
sfmg = sum(dout * out)
```

后续通常结合 softmax 概率 `P` 和 `dP` 推导：

```text
dS = P * (dP - row_sum)
```

本文件只负责提前算出 `row_sum`，不直接计算完整 `dS`。

---

## 4. 关键数据结构

### 4.1 `ReduceLastND`

```cpp
struct ReduceLastND {
    uint32_t originalSrcM;
    uint32_t originalSrcK;
    uint32_t srcM;
    uint32_t srcK;
    uint32_t dstM;
    uint32_t dstK;
};
```

它描述一个二维 ND tensor 沿最后一维 `K` 做归约时的形状：

```text
输入 src: (srcM, srcK)
输出 dst: (dstM, dstK)
```

在 sfmg 场景中：

```text
srcM = 当前处理的行数 nBurst
srcK = headdim 对齐后的 dAlign
dstK = 8
```

也就是每行归约成一个标量后，再广播为 8 个 float。

### 4.2 `SoftMaxShapeInfo`

```cpp
struct SoftMaxShapeInfo {
    uint32_t srcM{0};
    uint32_t srcK{0};
    uint32_t oriSrcM{0};
    uint32_t oriSrcK{0};
};
```

这是对外接口保留的形状参数结构。当前实现主要依赖 `LocalTensor::GetShapeInfo()` 推导真实 ND 形状，`softmaxShapeInfo` 本身更多是接口兼容参数。

---

## 5. 主要函数解析

### 5.1 `CustomAlignedReduceSumNDImpl`

作用：处理 `K` 维可以按 `FLOAT_REPEAT_SIZE` 完整切分的主体部分。

逻辑：

```text
src: (M, K)
K 被切成多个 256-float 片段

for 每个 256-float 片段:
    对每一行做 BlockReduceSum

把多个片段的结果加起来
最后再做一次 BlockReduceSum 得到每行总和
```

适合 `K` 较大且有完整 256-float 分段的情况。

### 5.2 `CustomReduceSumLastNDSplitImpl`

作用：处理 `K` 维尾块。

当：

```text
originalSrcK % FLOAT_REPEAT_SIZE != 0
```

说明最后一段不足 256 个 float，需要用 mask 控制只归约有效元素。

### 5.3 `CustomSingleBlockBroadCastImpl`

作用：把每行的单个归约标量广播成 `dstK` 个元素。

sfmg 场景中通常是：

```text
dstK = 8
```

所以：

```text
row_sum = s

输出:
[s, s, s, s, s, s, s, s]
```

实现上优先使用 `Brcb` 按 8 行一组广播；如果 `M` 维尾部不足 8 行，则使用标量读取加 `Duplicate` 逐行补齐。

### 5.4 `CustomReduceSumLastNDImpl`

作用：通用的 last-dim reduce + broadcast。

整体效果：

```text
dst[row, :] = sum(src[row, 0:K])
```

执行流程：

```text
1. 处理 K 维完整 256-float 分段
2. 处理 K 维尾块
3. 把每行归约标量广播为 dstK 个元素
```

### 5.5 `CustomSoftmaxGradFrontNDImpl`

这是核心计算函数。

数学含义：

```text
dst[row, :] = sum_j(gradTensor[row, j] * srcTensor[row, j])
```

它根据输入类型分成两大路径：

#### half / bf16 类路径

```text
srcTensor  -> cast float -> srcBuffer
gradTensor -> cast float -> gradBuffer
srcBuffer * gradBuffer -> dstBuffer
沿 K 维归约
广播
必要时 cast 回 T
```

#### float 路径

```text
srcTensor * gradTensor -> srcBuffer
沿 K 维归约
广播写入 dstTensor
```

在当前 FAG sfmg 调用中，调用方已经把 `dout/out` cast 成 `float`，因此主要走 float 路径。

### 5.6 `CustomSoftMaxGradTilingFunc`

作用：根据 UB 临时空间大小和输入形状生成 tiling 参数。

关键字段：

```text
reduceM      一次处理多少行
splitM       当前 M 维 tile 行数
splitK       K 维长度，通常是 headdim 对齐值
splitSize    splitM * splitK
rangeM       完整 M tile 数量
tailM        M 维尾行数
reduceK      输出每行的元素数，对 float 是 8
reduceSize   reduceM * reduceK
```

高性能 basic block 路径会额外调整 `reduceM`，要求它和 8 行基本块对齐，并避免单块数据量超过向量指令限制。

### 5.7 `SoftmaxGradFront`

这是对外入口。

```cpp
if ASCEND_IS_AIC{
    return;
}
```

说明它只在 AIV / Vector 核执行；如果当前是 AIC / Cube 侧，则直接返回。

---

## 6. 核心流程图

```text
输入:
  gradTensor: dout, shape = (M, K)
  srcTensor : out,  shape = (M, K)

          gradTensor              srcTensor
              │                      │
              └──────────┬───────────┘
                         ▼
                 逐元素乘 Mul
                         │
                         ▼
              product[row, j] = dout * out
                         │
                         ▼
                沿 K/headdim 做 reduce sum
                         │
                         ▼
                row_sum[row] = sum(product[row, :])
                         │
                         ▼
                广播为每行 8 个 float
                         │
                         ▼
               sfmg workspace / dstTensor
```

---

## 7. 例子一：单行 headdim=4 的简化计算

假设一行的：

```text
dout = [1, 2, 3, 4]
out  = [5, 6, 7, 8]
```

逐元素乘：

```text
dout * out = [5, 12, 21, 32]
```

沿 headdim 归约：

```text
sum = 5 + 12 + 21 + 32 = 70
```

输出到 sfmg workspace 时，为了 32B 对齐，写成 8 个 float：

```text
[70, 70, 70, 70, 70, 70, 70, 70]
```

虽然真实 headdim 可能是 64/128/256，这个例子展示的是同一行上的数学操作。

---

## 8. 例子二：M=10 行，按 8 行 basic block 处理

假设当前输入有 10 行，`isBasicBlock=true` 且形状满足高性能路径要求。

处理方式可以理解为：

```text
M = 10

前 8 行：
  使用 Brcb 高性能广播路径

后 2 行：
  不足 8 行，作为 tailM 或广播 tail 处理
  用标量读取 + Duplicate 补齐每行 8 个 float
```

示意图：

```text
行 0 ─┐
行 1  │
行 2  │
行 3  ├── 8 行一组：Brcb 广播
行 4  │
行 5  │
行 6  │
行 7 ─┘

行 8 ─┐
行 9 ─┴── 尾部不足 8 行：Duplicate 广播
```

这样兼顾了对齐场景下的性能和非整除场景下的正确性。

---

## 9. 与 `fag_epilogue_sfmg.hpp` 的关系

`fag_epilogue_sfmg.hpp` 负责：

```text
1. 从 GM 读取 dout 和 out
2. cast 到 float
3. 设置 LocalTensor ShapeInfo 为 (nBurst, dAlign)
4. 根据 nBurst/dAlign 判断 isBasicBlock
5. 调用 SoftmaxGradFront<float, true/false>
6. 把输出的 nBurst * 8 个 float 写到 sfmg workspace
```

`fag_sfmg.h` 负责第 5 步内部的实际行归约计算。

两者分工是：

```text
fag_epilogue_sfmg.hpp: 数据搬运、循环调度、workspace 写回
fag_sfmg.h          : dout*out 的行归约和广播
```

---

## 10. 总结

`fag_sfmg.h` 是一个专门服务于 softmax backward 前置归约的向量侧工具文件。

它完成的核心任务是：

```text
每行: sum(dout * out)
```

并把结果广播为每行 8 个 float，供后续 FAG vector epilogue 快速读取。

它的实现重点包括：

- 支持 float 和 half 类输入；
- half 类输入先 cast 到 float 再归约；
- 支持高性能 basic block 路径；
- 支持通用非对齐路径；
- 支持 K 维 256-float 整块归约；
- 支持 K 维尾块 mask 归约；
- 支持 M 维尾块处理；
- 输出按 32B 对齐布局组织。
