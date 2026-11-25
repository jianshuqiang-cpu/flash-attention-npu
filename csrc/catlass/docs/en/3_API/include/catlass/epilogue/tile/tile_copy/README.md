# tile_copy (Epilogue)

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Overview

Epilogue `tile_copy` is a template for aggregation and movement. It combines and references to basic movement templates such as `CopyGm2Ub` and `CopyUb2Gm`. It exposes child components as type members for use in block-level epilogue operations. It does not directly execute operators. Instead, all intermediate layouts and child component references are automatically derived from `GemmType`.

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [TileCopy](./tile_copy.md) | Non-TLA| Atlas A2 and Ascend 950| Basic aggregation, 2/3/4 operands|
| [TileCopyBf16](./tile_copy_bf16.md) | Non-TLA| Atlas A2 and Ascend 950| BF16 forced type specialization|
| [TileCopyPerTokenDequant](./tile_copy_per_token_dequant.md) | Non-TLA| Atlas A2| Per-token dequantization aggregation|
| [TileCopyW4A4Gemm](./tile_copy_w4a4_gemm.md) | Non-TLA| Atlas A2| W4A4 GEMM dequantization aggregation|
| [TileCopyDequantTla](./tile_copy_dequant_tla.md) | TLA | Atlas A2 and Ascend 950| TLA dequantization aggregation|

## Examples

### TileCopy

```cpp
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass::Epilogue::Tile;

using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DType = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopy<Arch::AtlasA2, CType, DType>;
using CopyC = typename Copy::CopyGmToUbC;
using CopyD = typename Copy::CopyUbToGmD;
```

### TileCopyPerTokenDequant

```cpp
using CType             = Gemm::GemmType<int32_t, layout::RowMajor>;
using ScaleType         = Gemm::GemmType<half, layout::RowMajor>;
using PerTokenScaleType = Gemm::GemmType<half, layout::ColumnMajor>;
using DType             = Gemm::GemmType<half, layout::RowMajor>;

using Copy = TileCopyPerTokenDequant<Arch::AtlasA2, CType, ScaleType, PerTokenScaleType, DType>;
```
