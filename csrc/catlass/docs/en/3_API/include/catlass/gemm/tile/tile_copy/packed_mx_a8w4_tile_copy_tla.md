# PackedMxA8W4TileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`PackedMxA8W4TileCopyTla` is derived from [PackedTileCopyTla](./packed_tile_copy_tla.md) and manages both MX Scale data movement and the movement of the A8W4 quantized B matrix. It is an extended variant of [PackedMxTileCopyTla](./packed_mx_tile_copy_tla.md) for the A8W4 (INT4 weight) scenario.

Key features:
- Matrix A is in FP8 format (`ElementA_`) and is transferred with MX Scale.
- Matrix B undergoes a type transformation: it starts as INT4 (represented by `ElementPrologueB_`, the type before the Prologue) and is converted to INT8 (represented by `ElementB_`, the type after the Prologue).
- `CopyL1ToL0B` is **overridden** to adapt to the B data type after the Prologue.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Referenced Tile Components

| Member Alias| Source|
| :------ | :------ |
| `CopyGmToL1A` ~ `CopyL1ToBT` (except L0B)| Inherited from `PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagPrologueB, ...>`|
| `CopyL1ToL0B` (overridden)| `TileCopyTla<ArchTag, TensorL1B, TensorL0B>` (type after the Prologue)|
| `CopyGmToL1MxScaleA` (new)| `TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>` |
| `CopyGmToL1MxScaleB` (new)| `TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>` |

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag: Arch::AtlasA2
    class ElementA_,                                                 // A matrix element type (FP8)
    class LayoutTagA,                                                // A GM layout tag
    class ElementPrologueB_,                                         // B pre-Prologue element type (INT4)
    class LayoutTagPrologueB,                                        // B pre-Prologue Global Memory (GM) layout tag
    class ElementB_,                                                 // B post-Prologue element type (INT8)
    class LayoutTagB,                                                // B post-Prologue GM layout tag
    class ElementMxScaleA_,                                          // A MX Scale element type
    class LayoutMxScaleA_,                                           // A MX Scale GM layout
    class ElementMxScaleB_,                                          // B MX Scale element type
    class LayoutMxScaleB_,                                           // B MX Scale GM layout
    class ElementC_,                                                 // C matrix element type
    class LayoutTagC,                                                // C GM layout tag
    class ElementBias = void,                                        // Bias element type
    bool ReluEnable_ = false,                                        // ReLU switch
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, // Dequantization granularity
    class L0CCopyMode = CopyToGM                                     // L0C → Dst transfer mode
>
struct PackedMxA8W4TileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA,
    ElementB_, LayoutTagPrologueB, ElementC_, LayoutTagC, ...>;
```

> **Note**: In the base class `PackedTileCopyTla`, `ElementB_` together with `LayoutTagPrologueB` specifies the B matrix type before the Prologue. `ElementB_` itself denotes the B matrix type after the Prologue, which is INT8 in this context.

## Overridden and Newly Added Members

```cpp
// Override B matrix layout derivation based on post-Prologue type.
using LayoutB     = detail::TagToLayout_t<ElementPrologueB_, LayoutTagPrologueB>;
using LayoutL1B   = detail::TagToLayout_t<ElementB_, LayoutTagL1B>;
using LayoutL0B   = detail::TagToLayout_t<ElementB_, LayoutTagL0B>;
using TensorL1B   = tla::Tensor<LocalTensor<ElementB_>, LayoutL1B, Coord<0,0>, A1>;
using TensorL0B   = tla::Tensor<LocalTensor<ElementB_>, LayoutL0B, Coord<0,0>, B2>;

// Override: L1 → L0B copy using post-Prologue B type
using CopyL1ToL0B = TileCopyTla<ArchTag, TensorL1B, TensorL0B>;

// Add MX Scale.
template <class TensorMxScaleA> using CopyGmToL1MxScaleA = TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;
template <class TensorMxScaleB> using CopyGmToL1MxScaleB = TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedMxA8W4TileCopyTla<
    Arch::AtlasA2,
    float8_e4m3_t, layout::RowMajor,          // A: FP8 RowMajor
    int8_t, layout::RowMajor,                 // B pre-Prologue type: INT4 (packed), RowMajor
    int8_t, layout::RowMajor,                 // B post-Prologue type: INT8, RowMajor
    float8_e8m0_t, layout::VectorLayout,      // MX Scale A
    float8_e8m0_t, layout::VectorLayout,      // MX Scale B
    half, layout::RowMajor>;                  // C: half RowMajor

typename TileCopy_::CopyGmToL1A         copyGmToL1A;
typename TileCopy_::CopyGmToL1B         copyGmToL1B;       // INT4 → GM→L1
typename TileCopy_::CopyL1ToL0B         copyL1ToL0B;       // INT8 L1 → L0B (post-Prologue)
typename TileCopy_::CopyGmToL1MxScaleA  copyGmToL1MxScaleA;
typename TileCopy_::CopyGmToL1MxScaleB  copyGmToL1MxScaleB;
typename TileCopy_::CopyL0CToGm         copyL0CToGm;
```
