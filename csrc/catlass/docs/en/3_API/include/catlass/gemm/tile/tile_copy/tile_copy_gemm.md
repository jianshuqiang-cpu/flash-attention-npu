# TileCopyGemm

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyGemm` is a data movement template collection specifically designed for General Matrix Multiply (GEMM). Its key distinction from the generic [TileCopy](./tile_copy.md) is the use of the `L1AndL0TypeSelectorGemm` selector, which automatically derives the L1 and L0 layouts for the A and B matrices. This allows the GM → L1 transfer of A and B to explicitly specify the destination L1 layout.

This template is intended for GEMM scenarios that require more complex layout transformations.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType, L1AType>` | Matrix A: Global Memory (GM) → L1 (with explicit L1 layout)|
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType, L1BType>` | Matrix B: GM → L1 (with explicit L1 layout)|
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType, L0AType>` | Matrix A: L1 → L0A|
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType, L0BType>` | Matrix B: L1 → L0B|
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>` | L0C→GM |

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag
    class AType,              // A matrix GmType
    class BType,              // B matrix GmType
    class CType,              // C matrix GmType
    class BiasType = void     // Bias GmType (optional, not using the L1/L0 selector)
>
struct TileCopyGemm;
```

## Member Type Derivation

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

## Differences from TileCopy

| Template| GM → L1 selector| L1 → L0 selector| Purpose|
| :------ | :------ | :------ | :------ |
| `TileCopy` | `L1ATypeSelector` / `L1BTypeSelector` | Same as GM → L1| General-purpose scenarios|
| `TileCopyGemm` | `L1AndL0TypeSelectorGemm` | `L1AndL0TypeSelectorGemm` | GEMM-specific, with explicit L0 derivation|

## Examples

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
