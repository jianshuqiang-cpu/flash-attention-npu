# CopyGm2UbAligned

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/copy_gm_to_ub.hpp)

[TOC]

## Function

`CopyGm2UbAligned` implements aligned data movement from global memory to Unified Buffer in the epilogue stage. Compared with `CopyGm2Ub`, it additionally handles scenarios where the stride is too large to be directly managed by `DataCopyPad`, automatically splitting into multiple `DataCopy` operations or performing row-wise movement.

- Applicability: Atlas A2
- Style: non-TLA, directly operating on `AscendC::LocalTensor`/`AscendC::GlobalTensor`
- Internal logic: contiguous memory →`DataCopy`; small stride → block `DataCopy`; large stride → row-wise movement

## Template Prototype

```cpp
template <
    class ArchTag,
    class GmType      // Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyGm2UbAligned;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Only `Arch::AtlasA2` has specialization.|
| `GmType` | GM data type. The layout is fixed to `RowMajor`.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,     // Destination Unified Buffer LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,    // Source global memory GlobalTensor
    layout::RowMajor const &layoutDst,                  // Destination Unified Buffer layout
    layout::RowMajor const &layoutSrc                   // Source global memory layout
)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination Unified Buffer LocalTensor|
| `srcTensor` | Source global memory GlobalTensor|
| `layoutDst` | RowMajor layout of the destination Unified Buffer|
| `layoutSrc` | RowMajor layout of the source global memory|

Internally, the optimal movement policy is automatically selected based on stride and dimension.
- No stride and no stride on the destination: directly `DataCopy(dst, src, rows * cols)`
- Small stride (< 65,536) and cols/blk < 65,536: block `DataCopy` with `DataCopyParams`
- Large stride: row-wise `DataCopy`

## Examples

```cpp
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Epilogue::Tile;

using Element = half;
uint32_t rows = 128;
uint32_t cols = 256;

auto layoutSrc = layout::RowMajor::MakeLayout<Element>(rows, cols);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(rows, cols);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using GmType = Gemm::GemmType<Element, layout::RowMajor>;
using CopyOp = CopyGm2UbAligned<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
