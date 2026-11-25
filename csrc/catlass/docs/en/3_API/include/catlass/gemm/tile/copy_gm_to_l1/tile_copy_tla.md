# TileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopyTla` is a TLA-style template for moving data from global memory to L1. Unlike `CopyGmToL1` (non-TLA), `TileCopyTla` encapsulates source and destination operands with `tla::Tensor`, automatically inferring movement parameters through TLA's layout/coord system, thereby simplifying the calling APIs.

The `Arch::AtlasA2` and `Arch::Ascend950` architectures are supported.

## Template Prototype

```cpp
template <
    class ArchTag,                                  // Architecture tag
    class TensorSrc,                                // Source operand TLA tensor type
    class TensorDst,                                // Destination operand TLA tensor type
    class Enable = void                             // SFINAE condition
>
struct TileCopyTla
```

The expected forms of `TensorSrc` and `TensorDst` are as follows:

```cpp
tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>   // Source
tla::Tensor<AscendC::LocalTensor<ElementDst>,  LayoutDst, CoordDst, AscendC::TPosition::A1>   // Destination
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`|
| `TensorSrc` | Source TLA tensor, which encapsulates global memory GlobalTensor, layout, coord, and TPosition::GM|
| `TensorDst` | Destination TLA tensor, which encapsulates L1 LocalTensor, layout, coord, and TPosition::A1|
| `Enable` | SFINAE condition, which restricts valid layout combinations through `std::enable_if_t`|

## Partial Specialization Implementation

### Atlas A2 Partial Specialization

All partially specialized `Enable` conditions are restricted by `std::enable_if_t<cond>`.

| Source Layout Condition| Destination Layout Condition| Description|
| :------ | :------ | :------ |
| `isRowMajor<LayoutSrc>` | `iszN<ElementDst, LayoutDst>` | RowMajor → zN |
| `isColumnMajor<LayoutSrc>` | `isnZ<ElementDst, LayoutDst>` | ColumnMajor → nZ|
| `iszN<ElementSrc, LayoutSrc>` | `iszN<ElementDst, LayoutDst>` | zN → zN (format preserved)|
| `isnZ<ElementSrc, LayoutSrc>` | `isnZ<ElementDst, LayoutDst>` | nZ → nZ (format preserved)|

### Ascend 950 Partial Specialization

| Source Layout Condition| Destination Layout Condition| Description|
| :------ | :------ | :------ |
| `isRowMajor<LayoutSrc>` | `iszN<ElementDst, LayoutDst>` | RowMajor → zN|
| `iszN<ElementSrc, LayoutSrc>` | `iszN<ElementDst, LayoutDst>` | zN → zN (format preserved)|
| `isColumnMajor<LayoutSrc>` | `isnZ<ElementDst, LayoutDst>` | ColumnMajor → nZ|
| `isnZ<ElementSrc, LayoutSrc>` | `isnZ<ElementDst, LayoutDst>` | nZ → nZ (format preserved)|
| `isVector<LayoutSrc>` | `isVector<LayoutDst>` | Vector → Vector (format preserved)|
| `isMxScaleForRowMajorA<fp8_e8m0_t, LayoutSrc>` | `isMxScaleForzZ<fp8_e8m0_t, LayoutDst>` | MX Scale RowMajor A → zZ|
| `isMxScaleForColumnMajorA<fp8_e8m0_t, LayoutSrc>` | `isMxScaleForzZ<fp8_e8m0_t, LayoutDst>` | MX Scale ColumnMajor A → zZ|
| `isMxScaleForRowMajorB<fp8_e8m0_t, LayoutSrc>` | `isMxScaleFornN<fp8_e8m0_t, LayoutDst>` | MX Scale RowMajor B → nN|
| `isMxScaleForColumnMajorB<fp8_e8m0_t, LayoutSrc>` | `isMxScaleFornN<fp8_e8m0_t, LayoutDst>` | MX Scale ColumnMajor B → nN|

## APIs

### Basic APIs (Atlas A2 All Partial Specialization + Ascend 950 zN/nZ/Vector/MX Scale)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,     // Destination TLA tensor
    TensorSrc const &srcTensor      // Source TLA tensor
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination TLA tensor (L1, TPosition::A1)|
| `srcTensor` | Source TLA tensor (global memory, TPosition::GM)|

### Extended APIs (Ascend 950 RowMajor/ColumnMajor Partial Specialization)

On Ascend 950, the RowMajor → zN and ColumnMajor → nZ partial specializations additionally support multi-matrix movement parameters.

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,         // Destination TLA tensor
    TensorSrc const &srcTensor,         // Source TLA tensor
    uint32_t ndNum = 1,                 // Number of ND matrices
    uint32_t srcNdMatrixStride = 0,     // Stride between source ND matrices
    uint32_t dstNzMatrixStride = 0      // Stride between destination matrices
)
```

| Parameter| Description|
| :------ | :------ |
| `ndNum` | Number of ND matrices to be continuously moved. The default value is 1.|
| `srcNdMatrixStride` | Stride between adjacent ND matrices on the source side. The default value is 0.|
| `dstNzMatrixStride` | Stride between adjacent matrices on the destination side. The default value is 0.|

## Examples

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

// Instantiation and call (SFINAE automatically matches the partial specialization based on the source/destination layout trait.)
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
