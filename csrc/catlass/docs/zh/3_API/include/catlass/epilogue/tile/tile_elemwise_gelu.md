# TileElemWiseGelu

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_gelu.hpp)

[TOC]

## 功能说明

`TileElemWiseGelu` 实现 epilogue 阶段的 GELU（Gaussian Error Linear Unit）激活函数，对 UB 上的输入 Tensor 逐元素计算 GELU 并输出。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- GELU 近似公式：`x / (1 + e^(-1.5957691 * 0.044715 * (x/0.044715 + x^3)))`
- 实现通过组合 `Mul`、`Axpy`、`Muls`、`Exp`、`Adds`、`Div` 指令完成

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ComputeType_,       // 计算数据类型（含 Element）
    uint32_t COMPUTE_LENGTH_  // 计算长度（element 个数）
>
struct TileElemWiseGelu;
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
| `dstLocal` | 目标 UB Tensor（被覆盖），存放 GELU(x) 结果 |
| `srcLocal` | 源 UB Tensor，输入值 x                      |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using GeluOp = TileElemWiseGelu<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> srcLocal;

GeluOp geluOp;
geluOp(dstLocal, srcLocal);
```
