# CopyGm2Ub/TileCopyTla (GM → UB)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_ub.hpp)

[TOC]

## Overview

The GM → UB module moves data from global memory to Unified Buffer. For one-dimensional vector data in `VectorLayout`, use `CopyGm2Ub`. For two-dimensional matrix data in `RowMajor`, use `TileCopyTla`.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## API List

| API | Style| Applicable Hardware| Layout| Description|
| :------ | :------ | :------ | :------ | :------ |
| [CopyGm2Ub](./copy_gm_to_ub.md) | Non-TLA| Atlas A2| VectorLayout| Global memory one-dimensional vector → Unified Buffer|
| [TileCopyTla](./tile_copy_tla.md) | TLA| Atlas A2| RowMajor| GM RowMajor → UB RowMajor|

## Examples

### Non-TLA

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"

using CopyOp = CopyGm2Ub<Arch::AtlasA2,
    Gemm::GemmType<half, layout::VectorLayout>>;

auto layoutSrc = layout::VectorLayout(len);
auto layoutDst = layout::VectorLayout(len);

CopyOp copyOp;
copyOp(dstUB, srcGm, layoutDst, layoutSrc);
```

### TLA

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"
#include "tla/tensor.hpp"

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto srcTensor = tla::MakeTensor(srcGm, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstUB, dstLayout, Arch::PositionUB{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

## Template Selection Guide

| Scenario| Recommendation| Style|
| :------ | :------ | :------ |
| One-dimensional vector movement (Bias/Scale)| `CopyGm2Ub` | Non-TLA|
| Two-dimensional matrix movement| `TileCopyTla`| TLA|
