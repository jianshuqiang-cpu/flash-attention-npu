# TileCopy

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopy` 是 GEMM Tile 层搬运模板的**基础集合模板**，通过模板参数推导出所有搬运算子类型，包括 GM→L1、L1→L0A/B、L0C→GM、Bias 搬运等。非 TLA 风格。

该模板不做任何实际计算，仅提供 `using` 类型别名供 blockMmad 组装使用。

## 引用的 Tile 组件

| 成员别名         | 引用的底层模板                                          | 说明                 |
| :--------------- | :------------------------------------------------------ | :------------------- |
| `CopyGmToL1A`    | `CopyGmToL1<ArchTag, AType>`                            | A 矩阵 GM→L1         |
| `CopyGmToL1B`    | `CopyGmToL1<ArchTag, BType>`                            | B 矩阵 GM→L1         |
| `CopyL1ToL0A`    | `CopyL1ToL0A<ArchTag, L1AType>`                         | A 矩阵 L1→L0A        |
| `CopyL1ToL0B`    | `CopyL1ToL0B<ArchTag, L1BType>`                         | B 矩阵 L1→L0B        |
| `CopyL0CToGm`    | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>`       | L0C→GM               |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, GMBiasType, L1BiasType>` 或 `void` | Bias GM→L1（条件性） |
| `CopyL1ToBT`     | `CopyL1ToBT<ArchTag, L1BiasType, L0BiasType>` 或 `void` | Bias L1→BT（条件性） |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签：Arch::AtlasA2 或 Arch::Ascend950
    class AType,              // A 矩阵 GmType
    class BType,              // B 矩阵 GmType
    class CType,              // C 矩阵 GmType
    class BiasType = void     // Bias GmType（可选）
>
struct TileCopy;
```

## 成员类型推导

```cpp
using ElementA = typename AType::Element;
using ElementB = typename BType::Element;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

using CopyGmToL1A = CopyGmToL1<ArchTag, AType>;
using CopyGmToL1B = CopyGmToL1<ArchTag, BType>;
using CopyL1ToL0A = CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
using CopyL1ToL0B = CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
using CopyGmToL1Bias = std::conditional_t<
    std::is_same_v<BiasType, void>, void,
    Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType,
                           typename BiasTypeSelector::L1BiasType>>;
using CopyL1ToBT = std::conditional_t<
    std::is_same_v<BiasType, void>, void,
    Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType,
                           typename BiasTypeSelector::L0BiasType>>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using TileCopy_ = Tile::TileCopy<Arch::AtlasA2, AType, BType, CType>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm   copyL0CToGm;
```
