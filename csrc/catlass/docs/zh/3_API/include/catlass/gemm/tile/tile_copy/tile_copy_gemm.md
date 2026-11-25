# TileCopyGemm

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyGemm` 是 GEMM 专用的搬运模板集合，与 [TileCopy](./tile_copy.md) 的关键区别在于使用 `L1AndL0TypeSelectorGemm` 选择器自动推导 L1 和 L0 布局。这使得 A/B 矩阵的 GM→L1 可以显式指定目标 L1 布局。

适用于 GEMM 场景中布局转换需求更复杂的场合。

## 引用的 Tile 组件

| 成员别名      | 引用的底层模板                                    | 说明                         |
| :------------ | :------------------------------------------------ | :--------------------------- |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType, L1AType>`             | A 矩阵 GM→L1（显式 L1 布局） |
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType, L1BType>`             | B 矩阵 GM→L1（显式 L1 布局） |
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType, L0AType>`          | A 矩阵 L1→L0A                |
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType, L0BType>`          | B 矩阵 L1→L0B                |
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>` | L0C→GM                       |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签
    class AType,              // A 矩阵 GmType
    class BType,              // B 矩阵 GmType
    class CType,              // C 矩阵 GmType
    class BiasType = void     // Bias GmType（可选，未使用 L1/L0 选择器）
>
struct TileCopyGemm;
```

## 成员类型推导

```cpp
using ElementA = typename AType::Element;
using ElementB = typename BType::Element;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

using L1AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1AType;
using L1BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1BType;
using L0AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0AType;
using L0BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0BType;

using CopyGmToL1A = CopyGmToL1<ArchTag, AType, L1AType>;
using CopyGmToL1B = CopyGmToL1<ArchTag, BType, L1BType>;
using CopyL1ToL0A = CopyL1ToL0A<ArchTag, L1AType, L0AType>;
using CopyL1ToL0B = CopyL1ToL0B<ArchTag, L1BType, L0BType>;
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
```

## 与 TileCopy 的区别

| 模板           | GM→L1 选择器                          | L1→L0 选择器              | 用途                    |
| :------------- | :------------------------------------ | :------------------------ | :---------------------- |
| `TileCopy`     | `L1ATypeSelector` / `L1BTypeSelector` | 同 L1                     | 通用场景                |
| `TileCopyGemm` | `L1AndL0TypeSelectorGemm`             | `L1AndL0TypeSelectorGemm` | GEMM 专用，显式 L0 推导 |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using TileCopy_ = Tile::TileCopyGemm<Arch::AtlasA2, AType, BType, CType>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;  // AType → L1AType
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;  // L1AType → L0AType
```
