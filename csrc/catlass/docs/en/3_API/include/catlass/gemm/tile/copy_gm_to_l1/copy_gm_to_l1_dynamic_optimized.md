# CopyGmToL1DynamicOptimized

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1DynamicOptimized` is a non-TLA template for GM-to-L1 data movement. Compared to `CopyGmToL1`, this template dynamically selects the optimal movement strategy at runtime based on the tile shape:

- When the number of rows or columns in a matrix is less than or equal to 16, the row-by-row or column-by-column strided `DataCopy` interface is used to avoid `Nd2Nz` instruction overhead.
- For larger matrices, the template uses the `Nd2Nz` instruction for efficient movement.

For scenarios where the format remains unchanged, such as zN → zN and nZ → nZ, it directly inherits from the corresponding `CopyGmToL1` specializations.

The `Arch::AtlasA2` and `Arch::Ascend950` architectures are supported.

## Template Prototype

```cpp
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand in Global Memory
    class L1Type = void     // GEMM type of the operand in L1 (**void** by default)
>
struct CopyGmToL1DynamicOptimized
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`|
| `GmType` | GEMM type of the source operand in Global Memory|
| `L1Type` | GEMM type of the destination operand in L1, which defaults to `void`|

## Partial Specialization Implementation

### Atlas A2 Partial Specialization

| GmType | Destination Layout| Implementation Method|
| :------ | :------ | :------ |
| `GemmType<Element, RowMajor>` | `zN` | Independent implementation, with dynamic policy selection|
| `GemmType<Element, ColumnMajor>` | `nZ` | Independent implementation, with dynamic policy selection|
| `GemmType<Element, zN>` | `zN` | Inherited from `CopyGmToL1<AtlasA2, GmType<Element, zN>>`|
| `GemmType<Element, nZ>` | `nZ` | Inherited from `CopyGmToL1<AtlasA2, GmType<Element, nZ>>`|
| `GemmType<Element, PaddingRowMajor>` | `zN` | Inherited from `CopyGmToL1<AtlasA2, GmType<Element, PaddingRowMajor>>`|
| `GemmType<Element, PaddingColumnMajor>` | `nZ` | Inherited from `CopyGmToL1<AtlasA2, GmType<Element, PaddingColumnMajor>>`|

### Ascend 950 Partial Specialization

| GmType | Destination Layout| Implementation Method|
| :------ | :------ | :------ |
| `GemmType<Element, RowMajor>` | `zN` | Independent implementation, with dynamic policy selection|
| `GemmType<Element, ColumnMajor>` | `nZ` | Independent implementation, with dynamic policy selection|
| `GemmType<Element, zN>` | `zN` | Inherited from `CopyGmToL1<Ascend950, GmType<Element, zN>>`|
| `GemmType<Element, nZ>` | `nZ` | Inherited from `CopyGmToL1<Ascend950, GmType<Element, nZ>>`|

## APIs

All partial specializations use the same API.

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

## Examples

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// Define the GEMM type on global memory.
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;
// Define the GEMM type on the L1.
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
