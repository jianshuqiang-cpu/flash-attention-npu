# copy_ub_to_gm

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## Overview

The `copy_ub_to_gm` module implements data movement from Unified Buffer to global memory in the epilogue stage and writes the epilogue computational result back to global memory.

UB → GM in TLA style is provided as an independent module [CopyUb2GmTla](../copy_ub_to_gm_tla.md).

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyUb2Gm](./copy_ub_to_gm.md) | Non-TLA| Atlas A2 and Ascend 950| Basic UB → GM movement (RowMajor/VectorLayout)|
| [CopyUb2GmAligned](./copy_ub_to_gm_aligned.md) | Non-TLA| Atlas A2| Aligned optimized movement (automatically handling large stride scenarios)|

## Applicable Hardware

| Hardware Model| CopyUb2Gm | CopyUb2GmAligned |
| :------ | :------ | :------ |
| Atlas A2| RowMajor, VectorLayout | RowMajor |
| Ascend 950 | RowMajor | - |

## Examples

### CopyUb2Gm

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagDst = layout::RowMajor;

uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = LayoutTagDst::MakeLayout<Element>(rows, cols);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagDst>;
using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
