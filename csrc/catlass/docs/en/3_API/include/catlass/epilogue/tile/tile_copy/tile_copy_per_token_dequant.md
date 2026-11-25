# TileCopyPerTokenDequant

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyPerTokenDequant` is a template for epilogue per-token dequantization movement and aggregation. In addition to `TileCopy`, it introduces `CopyPerTokenScale2Ub` to move the per-token scale to Unified Buffer for dequantization computation.

- Applicability: Atlas A2

## Template Prototype

```cpp
template <
    class ArchTag,
    class CType,                // int32_t accumulation result (RowMajor)
    class ScaleType,            // Per-channel scale (RowMajor)
    class PerTokenScaleType,    // Per-token scale (ColumnMajor)
    class DType                 // Dequantization target (RowMajor)
>
struct TileCopyPerTokenDequant;
```

## Member Types

| Member Type| Description|
| :------ | :------ |
| `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>` |
| `CopyGmToUbScale` | `CopyGm2Ub<Arch, ScaleType>` |
| `CopyGmToUbPerTokenScale` | `CopyPerTokenScale2Ub<Arch, PerTokenScaleType>` |
| `CopyUbToGmD` | `CopyUb2Gm<Arch, DType>` |

## Examples

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType             = Gemm::GemmType<int32_t, layout::RowMajor>;
using ScaleType         = Gemm::GemmType<half, layout::RowMajor>;
using PerTokenScaleType = Gemm::GemmType<half, layout::ColumnMajor>;
using DType             = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopyPerTokenDequant<Arch::AtlasA2, CType, ScaleType, PerTokenScaleType, DType>;

// Members:
// Copy::CopyGmToUbC            -> CopyGm2Ub<Arch, CType>
// Copy::CopyGmToUbScale        -> CopyGm2Ub<Arch, ScaleType>
// Copy::CopyGmToUbPerTokenScale -> CopyPerTokenScale2Ub<Arch, PerTokenScaleType>
// Copy::CopyUbToGmD            -> CopyUb2Gm<Arch, DType>
```
