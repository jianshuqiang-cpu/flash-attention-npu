# TileCopyTla (GM → UB)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_ub.hpp)

[TOC]

## Function

GM → UB partial specialization of `TileCopyTla` moves RowMajor two-dimensional matrix data from global memory to Unified Buffer (`VECCALC`) for access by the vector engine. Unlike the VectorLayout in [CopyGm2Ub](./copy_gm_to_ub.md), the TLA version supports the RowMajor two-dimensional layouts and performs row-by-row movement using `AscendC::DataCopyPad`.

> **Restriction**: This template supports only the Atlas A2 architecture. The source and destination layouts must be RowMajor.

## Template Prototype

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>>;
```

## Partial Specialization Implementation

| Architecture| Source Location| Destination Location| Layout Requirements| Movement Instruction|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| Global memory| VECCALC | RowMajor → RowMajor | `AscendC::DataCopyPad` |

Copy row by row, with each row length equal to `col * sizeof(ElementSrc)` bytes, for a total of `row` rows; the source stride corresponds to the destination stride's byte offset.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor (Unified Buffer, VECCALC, RowMajor)
    TensorSrc const &srcTensor     // Source tensor (global memory, RowMajor)
);
```

Static constraints:
- `TensorSrc::position == GM`; `TensorSrc::Layout` is RowMajor.
- `TensorDst::position == VECCALC`; `TensorDst::Layout` is RowMajor.

## Examples

```cpp
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementSrc = half;
using ElementDst = half;

const int M = 128;
const int K = 256;

auto srcLayout = tla::MakeLayout<ElementSrc, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, K);

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstUBTensor, dstLayout, Arch::PositionUB{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
