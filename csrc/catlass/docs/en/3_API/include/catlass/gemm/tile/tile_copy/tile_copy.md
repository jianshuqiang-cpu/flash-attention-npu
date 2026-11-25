# TileCopy

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopy` is the **fundamental aggregation template** for data movement at the General Matrix Multiply (GEMM) Tile layer. It derives, via template parameters, all data movement operator types, including movement from Global Memory (GM) to L1, L1 to L0A/B, and L0C to GM, as well as Bias movement, in a non-TLA style.

This template does not perform any actual computation. It only provides `using` type aliases for blockMmad assembly.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType>` | Matrix A: Global Memory (GM) → L1|
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType>` | Matrix B: GM → L1|
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType>` | Matrix A: L1 → L0A|
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType>` | Matrix B: L1 → L0B|
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>` | L0C→GM |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, GMBiasType, L1BiasType>` or `void`| Bias: GM → L1 (conditional)|
| `CopyL1ToBT` | `CopyL1ToBT<ArchTag, L1BiasType, L0BiasType>` or `void`| Bias: L1 → BT (conditional)|

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class AType,              // A matrix GmType
    class BType,              // B matrix GmType
    class CType,              // C matrix GmType
    class BiasType = void     // Bias GmType (optional)
>
struct TileCopy;
```

## Member Type Derivation

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

## Examples

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
