# CopyGmToL1GMMPTD

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1GMMPTD` is a non-TLA template for data movement from Global Memory (GM) to L1, specifically designed for the Permute-Transpose-DataCopy (PTD) stage in Group Matrix Multiplication (GMM). This template adds a special optimization for single-row matrices (using strided DataCopy), and provides an extended calling interface that allows manual stride specification.

The `Arch::AtlasA2` and `Arch::Ascend950` architectures are supported.

## Template Prototype

```cpp
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand in Global Memory
    class L1Type = void     // GEMM type of the operand in L1 (**void** by default)
>
struct CopyGmToL1GMMPTD
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`|
| `GmType` | GEMM type of the source operand in Global Memory|
| `L1Type` | GEMM type of the destination operand in L1, which defaults to `void`|

## Partial Specialization Implementation

| ArchTag | GmType | Destination Layout| Description|
| :------ | :------ | :------ | :------ |
| `Arch::AtlasA2` | `GemmType<Element, RowMajor>` | `zN` | RowMajor → zN, with single-row optimization|
| `Arch::Ascend950` | `GemmType<Element, RowMajor>` | `zN` | RowMajor → zN, with single-row optimization|

## APIs

### Basic APIs

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
| `dstTensor` | Destination L1 LocalTensor|
| `srcTensor` | Source global memory GlobalTensor|
| `layoutDst` | Layout description of the destination operand|
| `layoutSrc` | Layout description of the source operand|

### Extended APIs (Stride Specified Manually)

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
| `ndNum` | Number of ND matrices to be contiguously moved|
| `srcNdMatrixStride` | Stride between adjacent ND matrices at the source|
| `dstNzNStride` | Stride in the destination N direction (overwriting the default layout value)|
| `dstNzMatrixStride` | Stride between adjacent matrices at the destination (overwriting the default layout value)|
| `dstNzC0Stride` | Stride in the destination C0 direction (overwriting the default layout value)|

## Examples

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// In the GMM PTD scenario, only GmType needs to be specified. (The default value of L1Type is void, which is automatically deduced by the partial specialization.)
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
