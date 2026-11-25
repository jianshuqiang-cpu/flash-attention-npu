# PaddingPackedTileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`PaddingPackedTileCopyTla` is a Tensor Layout Abstraction (TLA) data movement template collection that supports padding-aware data movement. The key difference from [PackedTileCopyTla](./packed_tile_copy_tla.md) is that `CopyGmToL1A/B` uses `TileCopyTlaExt` (instead of `TileCopyTla`), which supports source-side PaddingRowMajor and PaddingColumnMajor layouts.

This template is designed for scenarios where matrix block alignment introduces padding. The `IS_PADDING_A` and `IS_PADDING_B` control whether to enable the padding layout tag.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `TileCopyTlaExt<ArchTag, TensorA, TensorL1A, PaddingTag/RowMajor, LayoutTagL1A>` | Matrix A: GM → L1 (Ext)|
| `CopyGmToL1B` | `TileCopyTlaExt<ArchTag, TensorB, TensorL1B, PaddingTag/ColumnMajor, LayoutTagL1B>` | Matrix B: GM → L1 (Ext)|
| `CopyL1ToL0A` | `TileCopyTla<ArchTag, TensorL1A, TensorL0A>` | Matrix A: L1 → L0A (TLA)|
| `CopyL1ToL0B` | `TileCopyTla<ArchTag, TensorL1B, TensorL0B>` | Matrix B: L1 → L0B (TLA)|
| `CopyL0CToDst` (Ascend 950) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>` | L0C→Dst |
| `CopyL0CToGm` (Atlas A2) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>` | L0C→GM |

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag
    class TensorA,                                                   // A tensor
    class LayoutTagA,                                                // A GM layout tag
    class TensorB,                                                   // B tensor
    class LayoutTagB,                                                // B GM layout tag
    class TensorC,                                                   // C tensor
    class LayoutTagC,                                                // C GM layout tag
    class TensorBias = void,                                         // Bias tensor
    class LayoutTagBias = void,                                      // Bias layout tag
    bool IS_PADDING_A = false,                                       // Whether matrix A requires padding.
    bool IS_PADDING_B = false                                        // Whether matrix B requires padding.
>
struct PaddingPackedTileCopyTla;
```

> **LayoutTagA/LayoutTagB constraints**: Only `layout::RowMajor` or `layout::ColumnMajor` is supported.

## Padding Logic

When `IS_PADDING_A = true`:
```cpp
using LayoutPaddingTagA = std::conditional_t<
    std::is_same_v<LayoutTagA, layout::RowMajor>,
    layout::PaddingRowMajor,
    layout::PaddingColumnMajor>;

using CopyGmToL1A = TileCopyTlaExt<ArchTag, TensorA, TensorL1A,
    LayoutPaddingTagA, LayoutTagL1A>;   // Ext + Padding
```

## Differences from PackedTileCopyTla

| Template| GM → L1 Operator| Source Padding| Template Parameter Style|
| :------ | :------ | :------ | :------ |
| `PackedTileCopyTla` | `TileCopyTla` | Not supported| Element + LayoutTag (non-Tensor)|
| `PaddingPackedTileCopyTla` | `TileCopyTlaExt` | Supported (IS_PADDING_A/IS_PADDING_B)| Tensor parameters (with Padding Layout)|

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm;

using ElementA = half;
using ElementB = half;
using ElementC = half;

// Matrix A after padding
auto layoutA = tla::MakeLayout<ElementA, layout::PaddingRowMajor>(M, K_padded);
auto tensorA = tla::MakeTensor(gmTensorA, layoutA, Arch::PositionGM{});

auto layoutB = tla::MakeLayout<ElementB, layout::ColumnMajor>(K_padded, N);
auto tensorB = tla::MakeTensor(gmTensorB, layoutB, Arch::PositionGM{});

auto layoutC = tla::MakeLayout<ElementC, layout::RowMajor>(M, N);
auto tensorC = tla::MakeTensor(gmTensorC, layoutC, Arch::PositionGM{});

using TileCopy_ = Tile::PaddingPackedTileCopyTla<
    Arch::AtlasA2,
    decltype(tensorA), layout::RowMajor,      // PaddingRowMajor → RowMajor
    decltype(tensorB), layout::ColumnMajor,
    decltype(tensorC), layout::RowMajor,
    void, void,
    true, false>;                              // IS_PADDING_A = true

typename TileCopy_::CopyGmToL1A copyGmToL1A;  // TileCopyTlaExt, PaddingRowMajor
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```
