# CopyPerTokenScale2Ub

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## Function

`CopyPerTokenScale2Ub` implements special movement of per-token scale from global memory to Unified Buffer. It moves scale data of shape (m,  1) from global memory to the first column of a Unified Buffer matrix of shape (m,  n), and applies padding along block boundaries.

Typical scenario: In per-token dequantization, the per-token scale is moved from global memory ColumnMajor to the first column of a Unified Buffer RowMajor matrix.

- Applicability: Atlas A2 (no architecture restriction, but with static layout assertions)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`/`AscendC::GlobalTensor`
- The global memory layout supports only `ColumnMajor`.
- Implemented by `AscendC::DataCopyPad` and padding

## Template Prototype

```cpp
template <
    class ArchTag,
    class GmType       // Gemm::GemmType<Element, layout::ColumnMajor>
>
struct CopyPerTokenScale2Ub;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `GmType` | GM data type, with a static assertion that the `Layout` is `ColumnMajor`.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // Destination Unified Buffer LocalTensor (RowMajor, m×n)
    AscendC::GlobalTensor<Element> const &srcTensor,    // Source global memory GlobalTensor (ColumnMajor, m×1)
    LayoutDst const &layoutDst,                         // Destination Unified Buffer RowMajor layout
    LayoutSrc const &layoutSrc                          // Source global memory ColumnMajor layout
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination Unified Buffer LocalTensor, with RowMajor layout. The scale value is written into the first column of each row and padding is performed.|
| `srcTensor` | Source global memory GlobalTensor, ColumnMajor layout (m, 1)|
| `layoutDst` | Destination Unified Buffer layout. `layoutDst.shape(1)` is used to compute dstStride.|
| `layoutSrc` | Source global memory layout. `layoutSrc.shape(0)` is m.|

Internally, the `DataCopyPad` padding parameter `isPad = true` ensures that the first-column data of each row is padded to a full block.

## Examples

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
using LayoutTagSrc = layout::ColumnMajor;
using LayoutTagDst = layout::RowMajor;

uint32_t m = 128;
uint32_t n = 256;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(m, 1);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(m, n);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;
using CopyOp = CopyPerTokenScale2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
