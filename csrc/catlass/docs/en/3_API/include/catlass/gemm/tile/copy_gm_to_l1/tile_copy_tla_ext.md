# TileCopyTlaExt

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopyTlaExt` is an extended version of `TileCopyTla`, also used for TLA-style data movement from global memory to L1. The main differences from `TileCopyTla` are:

1. **Template parameters**: `TileCopyTlaExt` matches partial specializations via explicit `LayoutTagSrc` and `LayoutTagDst` template parameters, rather than using `std::enable_if_t` + trait detection.
2. **API calls**: `operator()` of `TileCopyTlaExt` additionally accepts an `ActualShape` parameter, allowing the caller to specify the actual block shape to be moved (instead of using the tensor's full shape), which is useful for scenarios requiring padding or partial movement.

This template supports only the `Arch::AtlasA2` architecture.

## Template Prototype

```cpp
template <
    class ArchTag,          // Architecture tag
    class TensorSrc,        // Source operand TLA tensor type
    class TensorDst,        // Destination operand TLA tensor type
    class LayoutTagSrc,     // Source layout tag (explicitly specified)
    class LayoutTagDst      // Destination layout tag (explicitly specified)
>
struct TileCopyTlaExt
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Only `Arch::AtlasA2` is supported.|
| `TensorSrc` | Source TLA tensor, which encapsulates global memory GlobalTensor, layout, coord, and TPosition::GM|
| `TensorDst` | Destination TLA tensor, which encapsulates L1 LocalTensor, layout, coord, and TPosition::A1|
| `LayoutTagSrc` | Explicitly specified source layout tag, such as `layout::RowMajor` and `layout::PaddingRowMajor`|
| `LayoutTagDst` | Explicitly specified destination layout tag, such as `layout::zN` and `layout::nZ`|

## Partial Specialization Implementation

All partial specializations apply only to `Arch::AtlasA2`.

| LayoutTagSrc | LayoutTagDst | Description|
| :------ | :------ | :------ |
| `layout::RowMajor` | `layout::zN` | RowMajor → zN, supporting ActualShape|
| `layout::PaddingRowMajor` | `layout::zN` | PaddingRowMajor → zN, supporting ActualShape|
| `layout::ColumnMajor` | `layout::nZ` | ColumnMajor → nZ, supporting ActualShape|
| `layout::PaddingColumnMajor` | `layout::nZ` | PaddingColumnMajor → nZ, supporting ActualShape|
| `layout::zN` | `layout::zN` | zN → zN (format preserved), supporting ActualShape|
| `layout::nZ` | `layout::nZ` | nZ → nZ (format preserved), supporting ActualShape|

## APIs

All specializations use the same API.

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,     // Destination TLA tensor
    TensorSrc const &srcTensor,     // Source TLA tensor
    ActualShape actualShape         // Shape of the actual data block to be moved
)
```

`ActualShape` is defined as `tla::Shape<uint32_t, uint32_t>`.

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination TLA tensor (L1, TPosition::A1)|
| `srcTensor` | Source TLA tensor (global memory, TPosition::GM)|
| `actualShape` | Shape of the data block to be moved (number of rows and columns), which can be smaller than the full shape of the tensor|

## Comparison with TileCopyTla

| Feature| TileCopyTla| TileCopyTlaExt|
| :------ | :------ | :------ |
| Partial specialization matching mode| `std::enable_if_t` + trait detection| Explicit `LayoutTagSrc`/`LayoutTagDst`|
| Supported architecture| Atlas A2/Ascend 950| Atlas A2 only|
| ActualShape parameters| Not supported| Supported|
| Padding layout| Not supported| Supported (PaddingRowMajor/PaddingColumnMajor)|
| Scenario| General matrix multiplication tile movement| Partial movement or padding scenario|

## Examples

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
