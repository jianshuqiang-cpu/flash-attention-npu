# CopyGmToL1

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1` is a non-TLA template that moves data from global memory to local memory (L1). It moves tiles from GlobalTensor to LocalTensor while converting the data layout during the movement.

This template supports various source and destination layout combinations, covering both matrix multiplication (GEMM) and vector multiplication (GEMV) scenarios. The partial specialization implementation varies depending on the architecture.
- `Arch::AtlasA2`(ARCH 2201):[atlasa2/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)
- `Arch::Ascend950`(ARCH 3510):[ascend950/copy_gm_to_l1.hpp](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp)

## Template Prototype

```cpp
template <
    class ArchTag,          // Architecture tag, for example, Arch::AtlasA2 / Arch::Ascend950
    class GmType,           // GEMM type of the operand on global memory
    class L1Type = void     // GEMM type of the operand on L1 (void by default, meaning the type is automatically inferred through partial specialization)
>
struct CopyGmToL1
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag`| Architecture tag, which determines the set of hardware instructions to be used. The value can be `Arch::AtlasA2` or `Arch::Ascend950`.|
| `GmType`| GEMM type of the source operand on global memory, which encapsulates the data type and layout information.|
| `L1Type`| GEMM type of the destination operand on L1, which encapsulates the data type, layout, and TPosition information. The default value is `void`, which is automatically inferred by the partial specialization.|

## Partial Specialization Implementation

### Atlas A2 Partial Specialization

The following partial specialization applies to `Arch::AtlasA2`.

#### Simplified Version (`GmType` Specified Only, `L1Type` Automatically Inferred)

Only `GmType` (two parameters) needs to be specified. The destination `Layout` and `TPosition` are automatically inferred by partial specialization, eliminating redundant declarations and making this approach ideal for common movement scenarios. `RowMajor → zN` additionally provides an extended API for manually specifying the stride.

| Source Layout| Destination Layout| Description|
| :------ | :------ | :------ |
| RowMajor| zN| Dual calling APIs (basic + manual stride) included. For details, see the APIs below.|
| ColumnMajor| nZ| Commonly used for matrix B movement.|
| PaddingRowMajor| zN| RowMajor with padding, used for non-aligned matrix multiplication.|
| PaddingColumnMajor| nZ| ColumnMajor with padding, used for non-aligned matrix multiplication.|
| zN| zN| Keeps the zN format unchanged.|
| nZ| nZ| Keeps the nZ format unchanged.|

#### GEMM Scenario

| Source Layout| Destination Layout| Description|
| :------ | :------ | :------ |
| RowMajor| zN (A1)| Moves matrix A and converts it to zN format.|
| RowMajor| zZ (B1)| Moves matrix B and converts it to zZ format.|
| RowMajor| zN (B1)| Moves matrix B and converts it to zN format.|
| RowMajor| RowMajor (A1)| Keeps the RowMajor format unchanged.|
| ColumnMajor| nN (A1)| Moves matrix A and converts it to nN format.|
| ColumnMajor| nZ (A1)| Moves matrix A and converts it to nZ format.|
| ColumnMajor| nZ (B1)| Moves matrix B and converts it to nZ format.|
| ColumnMajor| nN (B1)| Moves matrix B and converts it to nN format.|

#### GEMV scenario

| Source Layout| Destination Layout| Description|
| :------ | :------ | :------ |
| VectorLayout| zN (A1)| Moves the vector and converts it to zN format.|
| VectorLayout (global memory)| VectorLayout (A1)| Moves the vector and retains the format.|

#### Convolution Scenario

| Source Layout| Destination Layout| Description|
| :------ | :------ | :------ |
| NDC1HWC0 (global memory)| NDC1HWC0| Keeps the format unchanged.|
| KDC1KHKWN1N0C0 (global memory)| nZ| Moves and converts to nZ format.|

### Ascend 950 Partial Specialization

The following partial specialization applies to `Arch::Ascend950`.

| Source Layout| Destination Layout| Description|
| :------ | :------ | :------ |
| RowMajor| zN| Simplified version, including dual calling APIs (basic + manual stride)|
| ColumnMajor| nZ| Simplified version|
| zN| zN| Keeps zN format unchanged.|
| nZ| nZ| Keeps the nZ format unchanged.|
| RowMajor| zZ (A1)| Dedicated for MX scale. Only the `fp8_e8m0_t` type is supported.|
| PaddingRowMajor| zN| RowMajor with padding|
| PaddingColumnMajor| nZ| ColumnMajor with padding|

## APIs

### Basic APIs (For All Partial Specialization)

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // Destination operand LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // Source operand GlobalTensor
    LayoutDst const &layoutDst,                       // Destination operand layout
    LayoutSrc const &layoutSrc                        // Source operand layout
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor`| Destination L1 LocalTensor|
| `srcTensor`| Source global memory GlobalTensor|
| `layoutDst`| Layout description of the destination operand, including the shape and stride information|
| `layoutSrc`| Layout description of the source operand, including the shape and stride information|

### Extended APIs (Stride Specified Manually)

Additional overloads are provided in the following partial specialization to manually specify the movement stride.

- `AtlasA2, RowMajor` (Simplified version)
- `AtlasA2, RowMajor → zN, A1` (General version)
- `Ascend950, RowMajor`

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // Destination operand LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // Source operand GlobalTensor
    LayoutDst const &layoutDst,                       // Destination operand layout
    LayoutSrc const &layoutSrc,                       // Source operand layout
    uint32_t ndNum,                                   // Number of ND matrices
    uint32_t srcNdMatrixStride,                       // Stride between source ND matrices
    uint32_t dstNzNStride,                            // Stride in the destination N direction
    uint32_t dstNzMatrixStride,                       // Stride between destination matrices
    uint32_t dstNzC0Stride                            // Stride in the destination C0 direction
)
```

| Parameter| Description|
| :------ | :------ |
| `ndNum`| Number of ND matrices to be contiguously moved|
| `srcNdMatrixStride`| Stride between adjacent ND matrices at the source|
| `dstNzNStride`| Stride in the destination N direction (overwriting the default layout value)|
| `dstNzMatrixStride`| Stride between adjacent matrices at the destination (overwriting the default layout value)|
| `dstNzC0Stride`| Stride in the destination C0 direction (overwriting the default layout value)|

## Examples

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
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementSrc>(row, col);
// Construct the zN layout on L1.
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementSrc> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// Instantiation and call
using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType, L1Type>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
