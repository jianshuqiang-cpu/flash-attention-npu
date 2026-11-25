# TileCopyTla (L1 → BT Partial Specialization)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_bt.hpp) (Ascend 950)

[TOC]

## Function

`TileCopyTla` is a general-purpose tile-level data movement template in the Tensor Layout Abstraction (TLA) style. The partial specialization defined in `copy_l1_to_bt.hpp` is specifically responsible for moving the Bias Table (1D vector) from L1 (A1 Buffer) to BT (Bias Table Buffer, C2 Buffer).

Unlike the [non-TLA-CopyL1ToBT](./copy_l1_to_bt.md), the TLA version encapsulates operands via `tla::Tensor`, allowing the TLA runtime to automatically deduce Layout, Shape, and Stride.

> **Note**: This specialization supports only the Ascend 950 architecture (`CATLASS_ARCH == 3510`). Atlas A2 does not provide a TLA-style L1-to-BT movement path.

## Template Prototype

`TileCopyTla` is defined in [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp).

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

The L1-to-BT partial specialization is matched via SFINAE under the following conditions: the source tensor has a `VectorLayout` (as recognized by the `isVector` trait), the destination tensor also has a `VectorLayout`, and their positions are `A1` and `C2` respectively.

## Partial Specialization Implementation

### Ascend 950

| Source Tensor| Destination Tensor| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| VectorLayout L1 (A1)| VectorLayout BT (C2)| `isVector<LayoutSrc> && isVector<LayoutDst>` | 1D vector copy using `AscendC::DataCopy`; automatic alignment for B32 data types based on blockLen|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`: source tensor (`tla::Tensor<LocalTensor, VectorLayout, Coord, A1>`) in L1
- `dstTensor`: destination tensor (`tla::Tensor<LocalTensor, VectorLayout, Coord, C2>`) in the BT buffer
- The element types of `srcTensor` and `dstTensor` can be different.

## Examples

### Ascend 950, TLA

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;

const uint32_t vecLen = 256;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::VectorLayout>(1, vecLen);
auto layoutDst = tla::MakeLayout<ElementDst, layout::VectorLayout>(1, vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstBTTensor, layoutDst, Arch::PositionBias{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

> **Note**: The second parameter in `tla::MakeLayout<Element, layout::VectorLayout>(1, vecLen)` indicates the vector length. TLA-side position mapping: `Arch::PositionL1` → A1, `Arch::PositionBias` → C2.
