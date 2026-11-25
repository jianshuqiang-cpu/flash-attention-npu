# TileCopySparseTla (L1 → L0A, Sparse)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0a.hpp)

[TOC]

## Function

The partial specialization defined by `TileCopySparseTla` in `copy_l1_to_l0a.hpp` is responsible for moving sparse A-matrix tile blocks from L1 (A1 Buffer) to L0A (A2 Buffer) in sparse General Matrix Multiply (GEMM) scenarios. The `LoadData3DParamsV2Pro` and `LoadData` instructions are used to implement zN → zZ format conversion.

> **Restriction**: This template supports only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) and is not supported on Ascend 950.

## Template Prototype

```cpp
template <
    class ArchTag,        // Architecture tag
    class TensorSrc,      // Source tensor (L1)
    class TensorDst,      // Destination tensor (L0A)
    class Enable = void   // SFINAE distribution
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Only `Arch::AtlasA2` is supported.|
| `TensorSrc` | Source tensor: `tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>`|
| `TensorDst` | Target tensor: `tla::Tensor<LocalTensor<Element>, Layout, Coord, A2>`|
| `Enable` | SFINAE condition, which automatically dispatches partial specialization based on layout|

## Partial Specialization Implementation

### Atlas A2

| Source Layout| Destination Layout| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| zN | zZ | `iszN<LayoutSrc> && iszZ<LayoutDst>` | LoadData3D v2 Pro, 16-aligned|

Hardware parameters: `HW_N0 = 16`, `HW_M0 = 16`. The matrix computation parameters are configured via `Load3DSetFMatrixCal`. `LoadData3DParamsV2Pro.extConfig` carries the M/K coordinate offsets and strides.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`: destination tensor on L0A (zZ format)
- `srcTensor`: source tensor on L1 (zN format)

## Examples

### zN L1 → zZ L0A (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

auto srcLayout = tla::MakeLayout<half, layout::zN>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::zZ>(M, K);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0ATensor;

auto srcTensor = tla::MakeTensor(srcL1Tensor, srcLayout, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0ATensor, dstLayout, Arch::PositionL0A{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
