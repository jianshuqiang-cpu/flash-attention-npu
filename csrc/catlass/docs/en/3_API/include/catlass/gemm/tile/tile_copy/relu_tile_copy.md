# ReluTileCopy

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`ReluTileCopy` is inherited from [TileCopy](./tile_copy.md) and **overrides** the `CopyL0CToGm` member type to enable FixPipe-based ReLU activation during L0C-to-GM movement.

All other member types (such as CopyGmToL1A/B and CopyL1ToL0A/B) are identical to those in the base class `TileCopy`.

## Referenced Tile Components

| Member Alias| Source|
| :------ | :------ |
| `CopyGmToL1A` ~ `CopyL1ToBT` | Inherited from `TileCopy<ArchTag, AType, BType, CType, BiasType>`|
| `CopyL0CToGm` (overridden)| `CopyL0CToGm<ArchTag, ElementAccumulator, CType, NO_QUANT, true>` |

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag
    class AType,              // A matrix GmType
    class BType,              // B matrix GmType
    class CType,              // C matrix GmType
    class BiasType = void     // Bias GmType (optional)
>
struct ReluTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType>;
```

## Overridden Members

```cpp
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType,
    ScaleGranularity::NO_QUANT, true>;  // ReluEnable = true
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using TileCopy_ = Tile::ReluTileCopy<Arch::AtlasA2, AType, BType, CType>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL0CToGm copyL0CToGm; // ReLU activation write-back
```
