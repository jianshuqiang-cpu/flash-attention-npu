# tile_copy (GEMV)

> [Code Location](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## Overview

GEMV `tile_copy` is an aggregation template for data movement. Based on the chip type, it provides two aggregation variants: AIV (bidirectional data movement between Global Memory and Unified Buffer) and AIC (data movement from Global Memory to L1, then to L0, and finally back to Global Memory).

## API List

| API | Chip Type| Applicable Hardware| Data Path| Description|
| :------ | :------ | :------ | :------ | :------ |
| [TileCopyGemvAiv](./tile_copy_gemv_aiv.md) | AIV | Atlas A2| Global Memory ↔ Unified Buffer| VecCopy + MatrixCopy |
| [TileCopyGemvAic](./tile_copy_gemv_aic.md) | AIC | Ascend 950| Global Memory → L1 → L0 → Global Memory| Reuses GEMM data movement components.|

## Examples

### TileCopyGemvAiv

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementA, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementA, layout::VectorLayout>;

using Copy = TileCopyGemvAiv<Arch::AtlasA2, AType, XType, YType>;
```

### TileCopyGemvAic

```cpp
using Copy = TileCopyGemvAic<Arch::Ascend950, AType, XType, YType>;
```
