# TileElemWiseSilu

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_silu.hpp)

[TOC]

## 功能说明

`TileElemWiseSilu` 实现 epilogue 阶段的 SiLU（Sigmoid Linear Unit）激活函数，也称为 Swish 函数。对 UB 上的输入 Tensor 逐元素计算 `x * sigmoid(x)` 并输出。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 公式：`SiLU(x) = x / (1 + e^(-x))`
- 实现通过组合 `Muls`（取负）、`Exp`、`Adds`、`Div` 指令完成

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ComputeType_,       // 计算数据类型（含 Element）
    uint32_t COMPUTE_LENGTH_  // 计算长度（element 个数）
>
struct TileElemWiseSilu;
```

| 模板参数          | 说明                                                    |
| :---------------- | :------------------------------------------------------ |
| `ArchTag_`        | 架构标签                                                |
| `ComputeType_`    | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `COMPUTE_LENGTH_` | 计算长度                                                |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &dstLocal,  // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementCompute> const &srcLocal   // 源 UB LocalTensor
)
```

| 参数       | 说明                                        |
| :--------- | :------------------------------------------ |
| `dstLocal` | 目标 UB Tensor（被覆盖），存放 SiLU(x) 结果 |
| `srcLocal` | 源 UB Tensor，输入值 x                      |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using SiluOp = TileElemWiseSilu<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> srcLocal;

SiluOp siluOp;
siluOp(dstLocal, srcLocal);
```
