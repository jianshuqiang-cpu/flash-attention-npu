# TileCopyTlaExt (UB → GM, PaddingRowMajor)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp)

[TOC]

## Function

The UB-to-GM partial specialization of `TileCopyTlaExt` is responsible for moving RowMajor 2D data from Unified Buffer (UB) (`VECCALC`) to Global Memory (GM), with the destination layout set to PaddingRowMajor. In PaddingRowMajor, the logical dimensions are the same as the original RowMajor, but the stride may be larger due to padding. This is commonly used for intermediate outputs that require alignment after matrix tiling.

Unlike the ordinary RowMajor destination of [TileCopyTla](./tile_copy_tla.md), this template targets the PaddingRowMajor layout.

> **Restriction**: Only the Atlas A2 architecture is supported.

## Template Prototype

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTlaExt<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    layout::RowMajor, layout::PaddingRowMajor>;
```

- `LayoutTagSrc = layout::RowMajor`: source layout tag (used solely for partial specialization dispatch; it is independent of the tensor's physical layout)
- `LayoutTagDst = layout::PaddingRowMajor`: destination layout tag

## Partial Specialization Implementation

| Architecture| Source Location| Destination Location| LayoutTagSrc | LayoutTagDst | Movement Instruction|
| :------ | :------ | :------ | :------ | :------ | :------ |
| Atlas A2| VECCALC | GM | RowMajor | PaddingRowMajor | `AscendC::DataCopyPad` |

Key differences from the common RowMajor destination version: The dimension calculation uses `tla::get<1, 1>(dstTensor.shape())` (number of logical rows) and `tla::get<1, 0>(dstTensor.shape())` (number of logical columns), and the offset is calculated based on the stride of PaddingRowMajor.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor (GM, PaddingRowMajor)
    TensorSrc const &srcTensor     // Source tensor (UB, VECCALC, RowMajor)
);
```

## Examples

```cpp
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<Element, layout::RowMajor>(M, N);
auto dstLayout = tla::MakeLayout<Element, layout::PaddingRowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcUBTensor, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

TileCopyTlaExt<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor),
    layout::RowMajor, layout::PaddingRowMajor> copyOp;
copyOp(dstTensor, srcTensor);
```
