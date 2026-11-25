# CopyGm2Ub

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_ub.hpp)

[TOC]

## Function

`CopyGm2Ub` is a template that moves one-dimensional vector data from global memory to Unified Buffer (`VECCALC`). It is often used in scenarios where auxiliary data such as bias and scale needs to be moved.

Different from the TLA-style [TileCopyTla](./tile_copy_tla.md), the non-TLA version supports only `VectorLayout` (one-dimensional vectors), and the movement instruction is `AscendC::DataCopyPad`.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag. Only Arch::AtlasA2 is supported.
    class GmType                      // Global memory data description, Gemm::GemmType<Element, layout::VectorLayout>
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub.");
};
```

## Partial Specialization Implementation

| Architecture| Layout| Movement Instruction| Description|
| :------ | :------ | :------ | :------ |
| Atlas A2| VectorLayout → VectorLayout| `AscendC::DataCopyPad`| One-dimensional vector, single-row copy, `blockLen=1`|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,      // Unified Buffer destination tensor (VECCALC)
    AscendC::GlobalTensor<Element> const &srcTensor,     // Global memory source tensor
    layout::VectorLayout const &layoutDst,               // Unified Buffer VectorLayout
    layout::VectorLayout const &layoutSrc                // Global memory VectorLayout
);
```

## Examples

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::VectorLayout>;

const uint32_t vecLen = 256;
auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::GlobalTensor<Element> srcGmTensor;
AscendC::LocalTensor<Element> dstUBTensor;

using CopyOp = CopyGm2Ub<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstUBTensor, srcGmTensor, layoutDst, layoutSrc);
```
