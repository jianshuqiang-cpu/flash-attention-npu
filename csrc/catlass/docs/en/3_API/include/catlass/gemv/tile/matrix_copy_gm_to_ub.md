# MatrixCopyGmToUB

> [Code Location](../../../../../../../include/catlass/gemv/tile/matrix_copy_gm_to_ub.hpp)

[TOC]

## Function

`MatrixCopyGmToUB` implements matrix data movement from Global Memory (GM) to Unified Buffer (UB) for General Matrix Multiplication (GEMV) scenarios. It automatically selects the optimal movement strategy (contiguous block, strided block, or row-by-row `DataCopy`) based on stride and element count.

- Applicability: Atlas A2
- Both RowMajor and ColumnMajor matrix layouts are supported.
- Fallback to row-by-row copy when `stride >= STRIDE_LIMIT(65536)`

## Template Prototype

```cpp
template <class ArchTag, class GmType>
struct MatrixCopyGmToUB;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `GmType` | `Gemm::GemmType<Element, RowMajor>` or `GemmType<Element, ColumnMajor>`|

## Partial Specialization Implementation

| Architecture| GmType | Movement Policy|
| :------ | :------ | :------ |
| Atlas A2| `RowMajor` | Three-level adaptive (contiguous block, strided block, or row-by-row)|
| Atlas A2| `ColumnMajor` | Three-level adaptive (contiguous block, strided block, or row-by-row)|

**Three-Level Movement Policies**

| Strategy| Trigger Condition| Method|
| :------ | :------ | :------ |
| Contiguous block| Length aligned with C0, stride aligned with C0, and stride < 65536| Single `DataCopy` (blockCount = m/n)|
| Strided block| Length aligned with C0 and stride × C0 < 65536| C0 `DataCopy` operations, each spaced by stride|
| Row-by-row| Fallback| Row-by-row `DataCopy`|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,            //UB destination
    AscendC::GlobalTensor<Element> srcTensor,           // GM source
    LayoutDst const &layoutDst,                         // UB layout (includes rounding/padding)
    LayoutSrc const &layoutSrc                          // GM layout (actual dimensions)
)
```

## Examples

### RowMajor

```cpp
#include "catlass/gemv/tile/matrix_copy_gm_to_ub.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using LayoutTagSrc = layout::RowMajor;

uint32_t m = 64, n = 128;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(m, n);
auto layoutDst = LayoutTagSrc::MakeLayout<Element>(m, n);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = MatrixCopyGmToUB<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
