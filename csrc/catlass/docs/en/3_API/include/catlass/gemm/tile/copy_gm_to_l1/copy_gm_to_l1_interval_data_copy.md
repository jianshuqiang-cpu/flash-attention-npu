# CopyGmToL1IntervalDataCopy

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1IntervalDataCopy` is a non-TLA template for moving data from the Global Memory (GM) to L1. Unlike `CopyGmToL1`, which uses the `Nd2Nz` instruction, this template uses the standard strided `DataCopy` interface to move data row by row, potentially achieving higher efficiency when the tile shape is "short and wide" or "tall and narrow".

Currently, only the `Arch::AtlasA2` architecture and the `half` data type are supported.

## Template Prototype

```cpp
template <
    class ArchTag,          // Architecture tag
    class GmType,           // GEMM type of the operand in Global Memory
    class L1Type = void     // GEMM type of the operand in L1 (**void** by default)
>
struct CopyGmToL1IntervalDataCopy
```

### Template Parameters

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Currently, only `Arch::AtlasA2` is supported.|
| `GmType` | GEMM type of the source operand in Global Memory|
| `L1Type` | GEMM type of the destination operand in L1, which defaults to `void`|

## Partial Specialization Implementation

All partial specializations apply only to `Arch::AtlasA2`, and the data type is fixed to `half`.

| GmType | Destination Layout| Description|
| :------ | :------ | :------ |
| `GemmType<half, RowMajor>` | `zN` | RowMajor → zN, row-by-row strided movement|
| `GemmType<half, PaddingRowMajor>` | `zN` | PaddingRowMajor → zN, row-by-row strided movement|
| `GemmType<half, ColumnMajor>` | `nZ` | ColumnMajor → nZ, column-by-column strided movement|
| `GemmType<half, PaddingColumnMajor>` | `nZ` | PaddingColumnMajor → nZ, column-by-column strided movement|

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
| `dstTensor` | Destination L1 LocalTensor, with element type `half`|
| `srcTensor` | Source GM GlobalTensor, with element type `half`|
| `layoutDst` | Layout description of the destination operand|
| `layoutSrc` | Layout description of the source operand|

## Examples

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

// Use strided DataCopy for row-by-row movement, applicable to short-fat/tall-skinny data blocks.
using CopyOp = CopyGmToL1IntervalDataCopy<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
