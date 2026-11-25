# CopyUb2GmAligned

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_ub_to_gm.hpp)

[TOC]

## Function

`CopyUb2GmAligned` implements aligned data movement from Unified Buffer to global memory in the epilogue stage. Compared with `CopyUb2Gm`, it additionally handles scenarios where the stride is too large to be directly managed by `DataCopyPad`, automatically splitting into multiple `DataCopy` operations or performing row-wise movement.

- Applicability: Atlas A2
- Style: non-TLA, directly operating on `AscendC::GlobalTensor`/`AscendC::LocalTensor`
- Internal logic: contiguous memory → `DataCopy`; small stride → block `DataCopy`; large stride → row-wise movement

## Template Prototype

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyUb2GmAligned;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Only `Arch::AtlasA2` has specialization.|
| `GmType` | GM data type. The layout is fixed to `RowMajor`.|

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,    // Destination global memory GlobalTensor
    AscendC::LocalTensor<Element> const &srcTensor,     // Source Unified Buffer LocalTensor
    layout::RowMajor const &layoutDst,                  // Destination global memory layout
    layout::RowMajor const &layoutSrc                   // Source Unified Buffer layout
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination global memory GlobalTensor|
| `srcTensor` | Source Unified Buffer LocalTensor|
| `layoutDst` | RowMajor layout of the destination global memory|
| `layoutSrc` | RowMajor layout of the source Unified Buffer|

Internally, the optimal movement policy is automatically selected based on stride and dimension: contiguous memory → direct `DataCopy`, small stride → block `DataCopy`, and large stride → row-wise movement.

## Examples

```cpp
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = layout::RowMajor::MakeLayout<Element>(rows, cols);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(rows, cols);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, layout::RowMajor>;
using CopyOp = CopyUb2GmAligned<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
