# Copy L1 To L0B Overview

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0b.hpp)

[TOC]

## Overview

The `copy_l1_to_l0b` module provides template classes that move tile blocks of the B matrix from L1 (Local Memory, referred to as B1 Buffer) to L0B (B2 Buffer), with support for conversion across multiple layouts. Based on the architectures, the implementation is split into two sets:

- **Atlas A2** (ARCH 2201): [atlasa2/copy_l1_to_l0b.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp)
- **Ascend 950** (ARCH 3510): [ascend950/copy_l1_to_l0b.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp)

The module provides two sets of APIs: a **non-TLA style** (directly operating on `LocalTensor`) and a **TLA style** (using `tla::Tensor` wrapper).

## API List

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL1ToL0B](./copy_l1_to_l0b.md) | Non-TLA| Atlas A2/Ascend 950| Basic L1 → L0B movement template, with support for conversion across multiple layouts|
| [TileCopyTla](./tile_copy_tla.md) | TLA | Atlas A2/Ascend 950| TLA-style L1 → L0B movement, which simplifies calling via the tla::Tensor wrapper|
| [CopyL1ToL0BSparseTla](./copy_l1_to_l0b_sparse_tla.md) | TLA | Atlas A2| Sparse L1 → L0B movement (Atlas A2 only), requiring an index tensor|

> **Note**: This module is not intended to be used directly in most cases. Instead, it is used as a member type of `CopyL1ToL0B` in [TileCopy](../tile_copy/README.md) and is automatically managed by [blockMmad](../../block/block_mmad.md). Explicit declaration is only required during implementation of custom kernel templates that require manual assembly.

## Applicable Hardware Models

| Hardware Model| Architecture ID| ARCH Macro| Supported Non-TLA Template| Supported TLA Template|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2 | `Arch::AtlasA2` | `CATLASS_ARCH == 2201` | CopyL1ToL0B | TileCopyTla / CopyL1ToL0BSparseTla |
| Ascend 950 | `Arch::Ascend950` | `CATLASS_ARCH == 3510` | CopyL1ToL0B | TileCopyTla |

### Architecture Differences

| Feature| Atlas A2| Ascend 950|
| :------ | :------ | :------ |
| Main movement direction| zZ → nZ (transposed), zN → nZ (transposed), nZ → nZ (direct)| zN → nZ (transposed), nZ → nZ (direct)|
| Basic movement instruction| LoadData2D/LoadData2dTranspose/LoadData3D | LoadData2DParamsV2 |
| l0Batch batch movement| Not supported| Supported (via `operator()` overload)|
| MX Scale floating-point quantization| Not supported| Supported (via `operator()` overload; B-side scale layout is nN)|
| B8/B4 narrow type| Supported (int8_t/float8_)| Supported (including MX Scale)|
| Sparse movement| Supported (CopyL1ToL0BSparseTla)| Not supported|
| GEMV-specific paths| Supported (zN → zN, nN → zN, nZ → zN)| Not supported|

## API Calling Examples

### Non-TLA style (CopyL1ToL0B)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;

uint32_t k = 256;
uint32_t n = 256;

auto layoutSrc = layout::zZ::MakeLayout<Element>(k, n);
auto layoutDst = layout::nZ::MakeLayout<Element>(k, n);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0BTensor;

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### TLA Style — zN → nZ Transposed Movement (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// iszN<LayoutSrc> && isnZ<LayoutDst> → Automatic matching for transposed partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style — nZ → nZ Direct Movement (Atlas A2, Transpose B)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// isnZ<LayoutSrc> && isnZ<LayoutDst> → Automatic matching for direct movement partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style — Ascend 950 Basic Movement (zN → nZ)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend950: zN L1 → nZ L0B
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style — Ascend 950 nZ → nZ Direct Movement

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### TLA Style — Ascend 950 l0Batch Batch Movement

```cpp
uint32_t l0Batch = 4;

copyOp(dstTensor, srcTensor, l0Batch);
```

### TLA Style — Ascend 950 MX Scale Movement (B Side)

```cpp
using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// Source data layout (L1 zN)
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(K, N);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// Destination data layout (L0B nZ, element type: mx_fp8)
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout (L1 nN, isMxScaleB=true on the B side, constructed using MakeMxScaleLayout)
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale overload: L1 zN source data + L1 nN scale → L0B nZ mx data
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

## Template Selection Guide

| Scenario| Recommended Template|
| :------ | :------ |
| GEMM tile-level L1 → L0B movement| `CopyL1ToL0B` (non-TLA) or `TileCopyTla` (TLA)|
| zZ → nZ transposed movement (Atlas A2)| `CopyL1ToL0B` or `TileCopyTla` (auto-matched)|
| zN → nZ transposed movement| `CopyL1ToL0B` or `TileCopyTla` (auto-matched)|
| nZ → nZ direct movement (Transpose B)| `CopyL1ToL0B` or `TileCopyTla` (auto-matched)|
| Ascend 950 multi-batch movement| `TileCopyTla` (l0Batch overload)|
| Ascend 950 MX floating-point quantization (B-side)| `TileCopyTla` (MX Scale overload, scale layout nN)|
| Atlas A2 sparse movement| `CopyL1ToL0BSparseTla` |
| Atlas A2 GEMV Scenario| `CopyL1ToL0B` (non-TLA)|
| TLA programming paradigm used| `TileCopyTla` (unified style)|
