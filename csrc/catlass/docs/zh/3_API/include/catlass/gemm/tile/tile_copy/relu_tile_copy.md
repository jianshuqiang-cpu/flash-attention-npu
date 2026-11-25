# ReluTileCopy

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`ReluTileCopy` 继承自 [TileCopy](./tile_copy.md)，**重写** `CopyL0CToGm` 成员类型，在 L0C→GM 搬运时启用 FixPipe 的 ReLU 激活功能。

所有其他成员类型（CopyGmToL1A/B、CopyL1ToL0A/B 等）与基类 `TileCopy` 完全一致。

## 引用的 Tile 组件

| 成员别名                     | 来源                                                              |
| :--------------------------- | :---------------------------------------------------------------- |
| `CopyGmToL1A` ~ `CopyL1ToBT` | 继承自 `TileCopy<ArchTag, AType, BType, CType, BiasType>`         |
| `CopyL0CToGm`（重写）        | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, NO_QUANT, true>` |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签
    class AType,              // A 矩阵 GmType
    class BType,              // B 矩阵 GmType
    class CType,              // C 矩阵 GmType
    class BiasType = void     // Bias GmType（可选）
>
struct ReluTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType>;
```

## 重写的成员

```cpp
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType,
    ScaleGranularity::NO_QUANT, true>;  // ReluEnable = true
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using TileCopy_ = Tile::ReluTileCopy<Arch::AtlasA2, AType, BType, CType>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;  // ReLU 激活写回
```
