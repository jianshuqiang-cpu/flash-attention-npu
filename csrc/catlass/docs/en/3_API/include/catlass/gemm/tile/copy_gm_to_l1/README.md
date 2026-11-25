# Copy Gm To L1 Overview

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## Overview

The `copy_gm_to_l1` module provides a template class for moving tiles from global memory to the local memory (L1) and supports conversion across multiple data layout formats. Its implementation is architecture-specific, with two variants:

- **Atlas A2** (ARCH 2201): [atlasa2/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)
- **Ascend 950** (ARCH 3510): [ascend950/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)

The module provides two sets of APIs: a **non-TLA style** (directly operating on `LocalTensor`/`GlobalTensor`) and a **TLA style** (encapsulated by `tla::Tensor`).

## API List

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyGmToL1](./copy_gm_to_l1.md)| Non-TLA| Atlas A2/Ascend 950| Basic GM → L1 movement template, which supports conversion across multiple layouts|
| [CopyGmToL1IntervalDataCopy](./copy_gm_to_l1_interval_data_copy.md) | Non-TLA| Atlas A2| Row-wise/Column-wise movement based on strided DataCopy, applicable to short-fat/tall-skinny data blocks|
| [CopyGmToL1GMMPTD](./copy_gm_to_l1_gmmptd.md)| Non-TLA| Atlas A2/Ascend 950| Dedicated movement in the GMM PTD scenario, including single-row optimization and manual stride APIs|
| [CopyGmToL1DynamicOptimized](./copy_gm_to_l1_dynamic_optimized.md)| Non-TLA| Atlas A2/Ascend 950| Runtime dynamic selection of a movement policy (strided DataCopy for small matrices and Nd2Nz for large ones)|
| [TileCopyTla](./tile_copy_tla.md)| TLA| Atlas A2/Ascend 950| TLA-style GM → L1 movement, simplified calling through `tla::Tensor` encapsulation|
| [TileCopyTlaExt](./tile_copy_tla_ext.md)| TLA| Atlas A2| TLA extended movement, supporting partial movement of `ActualShape` and padding layout|
| [TileCopySparseTla](./tile_copy_sparse_tla.md) | TLA | Atlas A2| Sparse GEMM GM → L1 movement, supporting RowMajor/ColumnMajor/zN/nZ → zN/nZ|
| [TileCopyFAQTla](./tile_copy_faq_tla.md) | TLA| Atlas A2| FlashAttention LoadQ movement, supporting three-dimensional multi-matrix GM → L1 zN conversion|

## Applicable Hardware Models

| Hardware Model| Architecture ID| ARCH Macro| Supported Non-TLA Template| Supported TLA Template|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| `Arch::AtlasA2`| `CATLASS_ARCH == 2201`| CopyGmToL1/CopyGmToL1IntervalDataCopy/CopyGmToL1GMMPTD/CopyGmToL1DynamicOptimized| TileCopyTla/TileCopyTlaExt|
| Ascend 950| `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyGmToL1/CopyGmToL1GMMPTD/CopyGmToL1DynamicOptimized| TileCopyTla |

## API Calling Examples

### Non-TLA Style (CopyGmToL1)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementSrc = half;
using ElementDst = half;

// Define the RowMajor data (matrix A) on global memory.
using GmType = Gemm::GemmType<ElementSrc, LayoutTagSrc>;
// Define the zN data on L1.
using L1Type = Gemm::GemmType<ElementDst, LayoutTagDst, AscendC::TPosition::A1>;

uint32_t row = 256;
uint32_t col = 256;

// Construct the RowMajor layout on global memory.
auto layoutSrc =LayoutTagSrc::MakeLayout<ElementSrc>(row, col);
// Construct the zN layout on L1.
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementSrc> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// Instantiation and call
using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### TLA Style (TileCopyTla)

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

// Create a layout using tla::MakeLayout (automatically infer the shape/stride based on the layout tag, element, and dimension).
auto layoutSrc = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

// Construct a TLA tensor using tla::MakeTensor.
AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;
auto srcTensor = tla::MakeTensor(srcGmTensor, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, layoutDst, Arch::PositionL1{});

// Instantiation and call
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style (TileCopyTlaExt)

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;
const uint32_t actualM = 128;
const uint32_t actualK = 128;

// Create a layout using tla::MakeLayout.
auto layoutSrc = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto layoutDst = tla::MakeLayout<half, layout::zN>(M, K);

// Construct a TLA tensor using tla::MakeTensor.
AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;
auto srcTensor = tla::MakeTensor(srcGmTensor, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, layoutDst, Arch::PositionL1{});

// Instantiate TileCopyTlaExt. (LayoutTagSrc/LayoutTagDst determines the movement policy, which is irrelevant to the tensor layout.)
TileCopyTlaExt<Arch::AtlasA2,
    decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::zN> copyOp;

// Specify the shape of the actual data block to be moved (which can be smaller than the full shape of the tensor).
tla::Shape<uint32_t, uint32_t> actualShape(actualM, actualK);
copyOp(dstTensor, srcTensor, actualShape);
```

### Dynamic Optimization Style (CopyGmToL1DynamicOptimized)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// Define the GEMM type on global memory.
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;
// Define the GEMM type on L1.
using L1Type = Gemm::GemmType<ElementDst, LayoutTagDst, AscendC::TPosition::A1>;

uint32_t row = 256;
uint32_t col = 256;

// Construct the layout.
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// Instantiate CopyGmToL1DynamicOptimized.
// Nd2Nz or strided DataCopy is automatically selected based on row or column.
using CopyOp = CopyGmToL1DynamicOptimized<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### GMM PTD Style (CopyGmToL1GMMPTD)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// In the GMM PTD scenario, only GmType needs to be specified. (The default value of L1Type is void, which is automatically inferred by the partial specialization.)
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// Construct the layout.
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// Basic call
using CopyOp = CopyGmToL1GMMPTD<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);

// Extended call: manually specifying the stride (multi-matrix movement scenario)
// copyOp(dstTensor, srcTensor, layoutDst, layoutSrc,
//        ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);
```

### Strided Movement Style (CopyGmToL1IntervalDataCopy)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;

// CopyGmToL1IntervalDataCopy supports only the half type and Atlas A2 architecture.
using GmType = Gemm::GemmType<half, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// Construct the layout.
auto layoutSrc = LayoutTagSrc::MakeLayout<half>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<half>(row, col);

AscendC::GlobalTensor<half> srcTensor;
AscendC::LocalTensor<half> dstTensor;

// Use strided DataCopy for row-wise movement, applicable to short-fat/tall-skinny data blocks.
using CopyOp = CopyGmToL1IntervalDataCopy<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

## Template Selection Guide

| Scenario| Recommended Template|
| :------ | :------ |
| General matrix multiplication tile movement| `CopyGmToL1` (non-TLA) or `TileCopyTla` (TLA)|
| Uncertain data block shape and runtime adaptation required| `CopyGmToL1DynamicOptimized` |
| Manual stride control required in the GMM PTD scenario| `CopyGmToL1GMMPTD` |
| Short-fat/tall-skinny data blocks (only the half type)| `CopyGmToL1IntervalDataCopy` |
| Partial movement or padding scenario| `TileCopyTlaExt` |
