# PackedTileCopyTlaToUB

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`PackedTileCopyTlaToUB` is inherited from [PackedTileCopyTla](./packed_tile_copy_tla.md) and **overrides** `CopyL0CToDst` to use `CopyL0CToUBTla`, directing the accumulated result to the Unified Buffer (UB) instead of Global Memory (GM). It supports UB split modes: `NO_SPLIT` and `SPLIT`.

This variant is intended for Ascend 950 architectures where the L0C result requires further post-processing by the Vector engine.

> **Restriction**: Only the Ascend 950 architecture (`CATLASS_ARCH == 3510`) is supported.

## Referenced Tile Components

| Member Alias| Source|
| :------ | :------ |
| `CopyGmToL1A` ~ `CopyL1ToBT` | Inherited from `PackedTileCopyTla<...>`|
| `CopyL0CToDst` (overridden)| `CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, ReluEnable>` |

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag: Arch::Ascend950
    class ElementA_,                                                 // A matrix element type
    class LayoutTagA,                                                // GM layout tag of matrix A
    class ElementB_,                                                 // B matrix element type
    class LayoutTagB,                                                // GM layout tag of matrix B
    class ElementC_,                                                 // C matrix element type
    class LayoutTagC,                                                // GM layout tag of matrix C
    class ElementBias = void,                                        // Bias element type
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,           // UB split mode
    bool ReluEnable = false,                                         // ReLU switch
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT // Dequantization granularity
>
struct PackedTileCopyTlaToUB : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ...>;
```

## Overridden Members

```cpp
template <class TensorC>
using CopyL0CToDst = CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode,
    DEQUANT_GRANULARITY, ReluEnable>;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedTileCopyTlaToUB<
    Arch::Ascend950,
    half, layout::RowMajor,
    half, layout::ColumnMajor,
    half, layout::RowMajor,
    void,
    CopyL0CToUBMode::SPLIT,      // UB split mode
    false,
    ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToDst copyL0CToUB; // → UB, not GM
```
