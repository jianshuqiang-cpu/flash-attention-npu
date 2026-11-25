# ConvTileCopy

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`ConvTileCopy` is a data movement aggregation template designed specifically for convolution (Conv) scenarios. It is structurally identical to [TileCopy](./tile_copy.md), with the only differences being the naming of template parameters and the dedicated use case—it is tailored exclusively for the Im2Col + GEMM pipeline in convolution operations.

BiasType is mandatory and does not default to `void` because Conv typically includes bias.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType>` | Matrix A: Global Memory (GM) → L1|
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType>` | Matrix B: GM → L1|
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType>` | Matrix A: L1 → L0A|
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType>` | Matrix B: L1 → L0B|
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>` | L0C→GM |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>` | Bias: GM → L1|
| `CopyL1ToBT` | `CopyL1ToBT<ArchTag, ...>` | Bias: L1 → BT|

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag
    class AType,              // A matrix GmType
    class BType,              // B matrix GmType
    class CType,              // C matrix GmType
    class BiasType            // Bias GmType (mandatory)
>
struct ConvTileCopy;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::NDC1HWC0>;
using BiasType = Gemm::GemmType<half, layout::VectorLayout>;

using TileCopy_ = Tile::ConvTileCopy<Arch::AtlasA2, AType, BType, CType, BiasType>;

typename TileCopy_::CopyGmToL1A   copyGmToL1A;
typename TileCopy_::CopyGmToL1B   copyGmToL1B;
typename TileCopy_::CopyGmToL1Bias copyGmToL1Bias;
typename TileCopy_::CopyL1ToBT    copyL1ToBT;
typename TileCopy_::CopyL0CToGm   copyL0CToGm;
```
