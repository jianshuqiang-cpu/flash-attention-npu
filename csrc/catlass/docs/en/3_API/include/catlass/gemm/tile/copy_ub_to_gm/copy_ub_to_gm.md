# CopyUb2Gm

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_ub_to_gm.hpp)

[TOC]

## Function

`CopyUb2Gm` is a template responsible for moving 2D matrix data from the Unified Buffer (UB, also referred to as `VECCALC`) to Global Memory (GM). It is commonly used after post-processing by the Vector engine to write the final results back to GM.

Unlike the TLA-style [TileCopyTla](./tile_copy_tla.md) or [TileCopyTlaExt](./tile_copy_tla_ext.md), this non-TLA version only supports the RowMajor layout and performs the movement using `AscendC::DataCopyPad`.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag. Only Arch::AtlasA2 is supported.
    class GmType                      // GM data description: Gemm::GemmType<Element, layout::RowMajor>
>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm.");
};
```

## Partial Specialization Implementation

| Architecture| Layout | Movement Instruction| Description|
| :------ | :------ | :------ | :------ |
| Atlas A2| RowMajor → RowMajor | `AscendC::DataCopyPad` | Row-by-row copy, with the number of rows determined by `shape(0)` and the row length determined by `shape(1)`|

Stride calculation:
- `srcStride = (layoutSrc.stride(0) - shape(1)) / ELE_NUM_PER_C0`
- `dstStride = (layoutDst.stride(0) - shape(1)) * sizeof(Element)`

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<Element> const &dstTensor,     // GM destination tensor
    AscendC::LocalTensor<Element> const &srcTensor,      // UB source tensor (VECCALC)
    layout::RowMajor const &layoutDst,                   // GM RowMajor layout
    layout::RowMajor const &layoutSrc                    // UB RowMajor layout
);
```

## Examples

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::RowMajor>;

const int M = 128;
const int N = 256;
auto layoutSrc = layout::RowMajor::MakeLayout<Element>(M, N);
auto layoutDst = layout::RowMajor::MakeLayout<Element>(M, N);

AscendC::LocalTensor<Element> srcUBTensor;
AscendC::GlobalTensor<Element> dstGmTensor;

using CopyOp = CopyUb2Gm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstGmTensor, srcUBTensor, layoutDst, layoutSrc);
```
