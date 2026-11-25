# TileElemwiseMul

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_mul.hpp)

[TOC]

## 功能说明

`TileElemwiseMul` 实现 epilogue 阶段的逐元素乘法操作，对 UB 上的两个输入 Tensor 做 element-wise Mul 并输出到目标 Tensor。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 通过 `AscendC::Mul` 指令完成计算

## 模板原型

```cpp
template <
    class ArchTag_,         // 架构标签
    class ComputeType_,     // 计算数据类型（含 Element）
    class TileShape_        // Tile 形状类型（含 COUNT）
>
struct TileElemwiseMul;
```

| 模板参数       | 说明                                                    |
| :------------- | :------------------------------------------------------ |
| `ArchTag_`     | 架构标签                                                |
| `ComputeType_` | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `TileShape_`   | Tile 形状类型，通过 `TileShape_::COUNT` 获取元素总数    |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,   // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementCompute> const &ubIn0,   // 源 UB LocalTensor 0
    AscendC::LocalTensor<ElementCompute> const &ubIn1    // 源 UB LocalTensor 1
)
```

| 参数    | 说明               |
| :------ | :----------------- |
| `ubOut` | 目标 UB Tensor     |
| `ubIn0` | 第一个源 UB Tensor |
| `ubIn1` | 第二个源 UB Tensor |

内部实现：`AscendC::Mul(ubOut, ubIn0, ubIn1, TileShape::COUNT)`

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_elemwise_mul.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using MulOp = TileElemwiseMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;

MulOp mulOp;
mulOp(ubOut, ubIn0, ubIn1);
```
