# TileCopyTla (UB → GM)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_ub_to_gm.hpp)

[TOC]

## Function

The Unified Buffer-to-Global Memory partial specialization of `TileCopyTla` is responsible for moving RowMajor 2D matrix data from UB (`VECCALC`) to GM, using `AscendC::DataCopyPad` to copy out row by row.

Unlike [TileCopyTlaExt](./tile_copy_tla_ext.md) with PaddingRowMajor destination layout, this template targets the standard RowMajor layout.

> **Restriction**: This template supports only the Atlas A2 architecture. The source and destination layouts must be RowMajor.

## Template Prototype

```cpp
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>>;
```

## Partial Specialization Implementation

| Architecture| Source Location| Destination Location| Layout Requirement| Movement Instruction|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| VECCALC | GM | RowMajor → RowMajor | `AscendC::DataCopyPad` |

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor (GM, RowMajor)
    TensorSrc const &srcTensor     // Source tensor (UB, VECCALC, RowMajor)
);
```

Static constraints:
- `TensorSrc::position == VECCALC`, and `TensorSrc::Layout` is RowMajor.
- `TensorDst::position == GM`, and `TensorDst::Layout` is RowMajor.

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
auto dstLayout = tla::MakeLayout<Element, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcUBTensor, srcLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
