# PackedMxTileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`PackedMxTileCopyTla` is inherited from [PackedTileCopyTla](./packed_tile_copy_tla.md) and extends it with **additional** data movement channels for MX Scale (micro-scaling) factors: `CopyGmToL1MxScaleA` and `CopyGmToL1MxScaleB`.

MX Scale is a block-wise scaling factor used in FP8 quantization, stored as `float8_e8m0_t`. During the Pack stage, the scales residing in Global Memory (GM) are transferred into L1 via TLA-based data movement (the A-side scale is converted to zZ layout, while the B-side scale is converted to nN layout). `PackedMxTileCopyTla` then manages the scheduling of subsequent tile-level operations using these scales.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Referenced Tile Components

| Member Alias| Source|
| :------ | :------ |
| `CopyGmToL1A` ~ `CopyL1ToBT` | Inherited from `PackedTileCopyTla<ArchTag, ...>`|
| `CopyGmToL1MxScaleA` (new)| `TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>` |
| `CopyGmToL1MxScaleB` (new)| `TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>` |

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag: Arch::AtlasA2
    class ElementA_,                                                 // A matrix element type (FP8)
    class LayoutTagA,                                                // A GM layout tag
    class ElementB_,                                                 // B matrix element type (FP8)
    class LayoutTagB,                                                // B GM layout tag
    class ElementMxScaleA_,                                          // A MX Scale element type (float8_e8m0_t)
    class LayoutMxScaleA_,                                           // A MX Scale GM layout
    class ElementMxScaleB_,                                          // B MX Scale element type (float8_e8m0_t)
    class LayoutMxScaleB_,                                           // B MX Scale GM layout
    class ElementC_,                                                 // C matrix element type
    class LayoutTagC,                                                // C GM layout tag
    class ElementBias = void,                                        // Bias element type
    bool ReluEnable_ = false,                                        // ReLU switch
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, // Dequantization granularity
    class L0CCopyMode = CopyToGM                                     // L0C → Dst transfer mode
>
struct PackedMxTileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
    ElementC_, LayoutTagC, ElementBias, ReluEnable_, DEQUANT_GRANULARITY, L0CCopyMode>;
```

## New Members

```cpp
using ElementMxScaleA = ElementMxScaleA_;
using ElementMxScaleB = ElementMxScaleB_;
using LayoutMxScaleA = LayoutMxScaleA_;
using LayoutMxScaleB = LayoutMxScaleB_;

// L1 layout is fixed to zZ(A) and nN(B).
using LayoutTagL1MxScaleA = layout::zZ;
using LayoutTagL1MxScaleB = layout::nN;
using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

template <class TensorMxScaleA>
using CopyGmToL1MxScaleA = TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

template <class TensorMxScaleB>
using CopyGmToL1MxScaleB = TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedMxTileCopyTla<
    Arch::AtlasA2,
    float8_e4m3_t, layout::RowMajor,       // A: FP8 RowMajor
    float8_e4m3_t, layout::ColumnMajor,    // B: FP8 ColumnMajor
    float8_e8m0_t, layout::VectorLayout,   // MX Scale A: RowMajor (on GM)
    float8_e8m0_t, layout::VectorLayout,   // MX Scale B: ColumnMajor (on the GM)
    half, layout::RowMajor>;              // C: half RowMajor

typename TileCopy_::CopyGmToL1A         copyGmToL1A;
typename TileCopy_::CopyGmToL1MxScaleA  copyGmToL1MxScaleA;  // Load A scale
typename TileCopy_::CopyGmToL1MxScaleB  copyGmToL1MxScaleB;  // Load B scale
typename TileCopy_::CopyL1ToL0A         copyL1ToL0A;
typename TileCopy_::CopyL0CToGm         copyL0CToGm;
```
