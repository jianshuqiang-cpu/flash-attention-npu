# CopyUb2Gm/TileCopyTla (UB → GM)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_ub_to_gm.hpp)

[TOC]

## Overview

The UB → GM movement module is responsible for moving data from the Unified Buffer (UB) back to Global Memory (GM). RowMajor-layout data can be moved using `CopyUb2Gm` (non-TLA), `TileCopyTla` (TLA, RowMajor), or `TileCopyTlaExt` (TLA, PaddingRowMajor).

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## API List

| API | Style| Applicable Hardware| Layout | Description|
| :------ | :------ | :------ | :------ | :------ |
| [CopyUb2Gm](./copy_ub_to_gm.md) | Non-TLA| Atlas A2| RowMajor | UB RowMajor → GM RowMajor |
| [TileCopyTla](./tile_copy_tla.md) | TLA | Atlas A2| RowMajor | TLA wrapper, RowMajor destination|
| [TileCopyTlaExt](./tile_copy_tla_ext.md) | TLA Ext | Atlas A2| PaddingRowMajor | TLA wrapper, PaddingRowMajor destination |

## Examples

### Non-TLA

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"

using CopyOp = CopyUb2Gm<Arch::AtlasA2,
    Gemm::GemmType<half, layout::RowMajor>>;

auto layoutSrc = layout::RowMajor::MakeLayout<half>(M, N);
auto layoutDst = layout::RowMajor::MakeLayout<half>(M, N);

CopyOp copyOp;
copyOp(dstGm, srcUB, layoutDst, layoutSrc);
```

### TLA

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "tla/tensor.hpp"

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto srcTensor = tla::MakeTensor(srcUB, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGm, dstLayout, Arch::PositionGM{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Ext (PaddingRowMajor)

```cpp
auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::PaddingRowMajor>(M, N);
auto srcTensor = tla::MakeTensor(srcUB, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGm, dstLayout, Arch::PositionGM{});

TileCopyTlaExt<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::PaddingRowMajor> copyOp;
copyOp(dstTensor, srcTensor);
```

## Template Selection Guide

| Scenario| Recommendation| Style|
| :------ | :------ | :------ |
| Common RowMajor write-back| `CopyUb2Gm` | Non-TLA|
| TLA-style RowMajor write-back| `TileCopyTla` | TLA |
| Post-padding RowMajor write-back| `TileCopyTlaExt` | TLA Ext |
