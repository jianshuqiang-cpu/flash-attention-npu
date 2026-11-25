# SparseTileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`SparseTileCopyTla` is a Tensor Layout Abstraction (TLA) data movement template collection specifically designed for sparse General Matrix Multiply (GEMM). All data movement operators in this collection follow the TLA style, including the sparse-specific `CopyL1ToL0BSparseTla` and `CopyL0CToGmSparseTla`.

The B matrix is loaded via `CopyGmToL1BIdx`, which additionally moves the index data (in CSR/COO format, type `int32_t`). The L1-to-L0B movement is handled by `CopyL1ToL0BSparseTla`, which uses the index tensor to carry out sparse decompression on-the-fly.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `TileCopySparseTla<ArchTag, TensorA, TensorL1A>` | Matrix A: GM → L1 (TLA)|
| `CopyGmToL1B` | `TileCopySparseTla<ArchTag, TensorB, TensorL1B>` | Matrix B: GM → L1 (TLA)|
| `CopyGmToL1BIdx` | `TileCopySparseTla<ArchTag, TensorIdx, TensorL1BIdx>` | B index: GM → L1 (TLA)|
| `CopyL1ToL0A` | `TileCopySparseTla<ArchTag, TensorL1A, TensorL0A>` | Matrix A: L1 → L0A (TLA)|
| `CopyL1ToL0B` | `CopyL1ToL0BSparseTla<ArchTag, ElementA, TensorL1B, TensorL0B, TensorL1BIdx>` | Matrix B: L1 → L0B (sparse TLA)|
| `CopyL0CToGm` | `CopyL0CToGmSparseTla<ArchTag, TensorL0C, TensorC>` | L0C → GM (sparse TLA)|

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag: Arch::AtlasA2
    class ElementA_,          // Element type of matrix A
    class LayoutTagA,         // GM layout tag of matrix A
    class ElementB_,          // Element type of matrix B
    class LayoutTagB,         // GM layout tag of matrix B
    class ElementC_,          // Element type of matrix C
    class LayoutTagC          // GM layout tag of matrix C
>
struct SparseTileCopyTla;
```

## Layout Deduction

```cpp
using LayoutTagL1A  = helper::L1ATypeSelector<GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
using LayoutTagL1B  = helper::L1BTypeSelector<GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
using LayoutTagL0A  = layout::zZ;
using LayoutTagL0B  = layout::nZ;
using LayoutTagL1BIdx = helper::L1BTypeSelector<GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
```

- Side A: GM LayoutTag → L1 LayoutTag → L0A zZ
- Side B: GM LayoutTag → L1 LayoutTag → L0B nZ
- B index: L1 LayoutTag that is the same as that of side B, type `int32_t`

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::SparseTileCopyTla<
    Arch::AtlasA2,
    half, layout::RowMajor,          // A: half RowMajor
    half, layout::ColumnMajor,       // B: half ColumnMajor
    half, layout::RowMajor>;         // C: half RowMajor

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1B     copyGmToL1B;
typename TileCopy_::CopyGmToL1BIdx  copyGmToL1BIdx;   // B matrix index movement
typename TileCopy_::CopyL1ToL0A     copyL1ToL0A;
typename TileCopy_::CopyL1ToL0B     copyL1ToL0B;       // Sparse decompression
typename TileCopy_::CopyL0CToGm     copyL0CToGm;
```
