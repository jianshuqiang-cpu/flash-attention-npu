# copy_gm_to_ub

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## Overview

The `copy_gm_to_ub` module implements data movement from global memory to Unified Buffer in the epilogue stage. It provides three struct variants: basic movement, per-token scale movement, and alignment-optimized movement.

GM → UB in TLA style is provided as an independent module [CopyGm2UbTla](../copy_gm_to_ub_tla.md).

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyGm2Ub](./copy_gm_to_ub.md) | Non-TLA| Atlas A2 and Ascend 950| Basic GM → UB movement (RowMajor/VectorLayout)|
| [CopyPerTokenScale2Ub](./copy_per_token_scale_to_ub.md) | Non-TLA| Atlas A2| Per-token scale dedicated movement (ColumnMajor → RowMajor, with padding)|
| [CopyGm2UbAligned](./copy_gm_to_ub_aligned.md) | Non-TLA| Atlas A2| Aligned optimized movement (automatically handling large stride scenarios)|

## Applicable Hardware

| Hardware Model| CopyGm2Ub | CopyPerTokenScale2Ub | CopyGm2UbAligned |
| :------ | :------ | :------ | :------ |
| Atlas A2| RowMajor, VectorLayout | ColumnMajor → RowMajor| RowMajor |
| Ascend 950 | RowMajor, VectorLayout | - | - |

## Examples

### CopyGm2Ub

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
