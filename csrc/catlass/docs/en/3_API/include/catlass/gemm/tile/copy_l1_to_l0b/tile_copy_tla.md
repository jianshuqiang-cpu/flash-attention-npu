# TileCopyTla (L1 → L0B Partial Specialization)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp) (Atlas A2)
> [Code Location](../../../../../../../../include/catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp) (Ascend 950)

[TOC]

## Function

`TileCopyTla` is a general-purpose tile-level data movement template in the Tensor Layout Abstraction (TLA) style. The partial specialization defined in `copy_l1_to_l0b.hpp` is specifically responsible for moving B-matrix tile blocks from L1 (B1 Buffer) to L0B (B2 Buffer).

Unlike [Non-TLA CopyL1ToL0B](./copy_l1_to_l0b.md), this TLA version encapsulates operands via `tla::Tensor`, allowing the TLA runtime to automatically deduce Layout, Shape, and Stride. The appropriate partial specialization is then matched via SFINAE (using traits such as `iszN` and `isnZ`).

## Template Prototype

`TileCopyTla` is defined in [tile_copy_tla.hpp](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp):

```cpp
template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla;
```

The partial specialization of L1 → L0B is matched through SFINAE: The position of the source tensor is `AscendC::TPosition::A1`, and the position of the destination tensor is `AscendC::TPosition::B2`.

## Partial Specialization Implementation

### Atlas A2

| Source Tensor| Destination Tensor| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| zN L1 | nZ L0B | `iszN<LayoutSrc> && isnZ<LayoutDst>` | Basic transposed copy (Transpose B)|
| zN L1 (int8_t) | nZ L0B (int8_t) | `iszN<int8_t, LayoutSrc> && isnZ<int8_t, LayoutDst>` | int8_t transpose (LoadDataWithTranspose)|
| zN L1 (float) | nZ L0B (float) | `iszN<float, LayoutSrc> && isnZ<float, LayoutDst>` | float transpose (LoadData3D + SetFmatrix)|
| nZ L1 | nZ L0B | `isnZ<LayoutSrc> && isnZ<LayoutDst>` | Non-transposed copy (direct copy)|

### Ascend 950

| Source Tensor| Destination Tensor| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| zN L1 (non-B8/B4)| nZ L0B (non-B8/B4)| `!is_one_of_v<Element, int8_t, float8_...> && iszN && isnZ` | Transposed copy. l0Batch overload is supported.|
| zN L1 (B8/B4)| nZ L0B (B8/B4)| `is_one_of_v<Element, int8_t, float8_...> && iszN && isnZ` | B8/B4 transposed copy. l0Batch overload and MX Scale overload are supported.|
| nZ L1 | nZ L0B | `isnZ<LayoutSrc> && isnZ<LayoutDst>` | Non-transposed copy (Transpose B). l0Batch overload and MX Scale overload are supported.|

> **Note**: The target layout of the TLA L1→L0B of Ascend 950 is nZ, and MX Scale floating-point quantization and l0Batch batch movement are supported.

## APIs

### Basic APIs (Common for Atlas A2 and Ascend 950)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor);
```

- `srcTensor`: source tensor (`tla::Tensor<LocalTensor, Layout, Coord, A1>`) in L1
- `dstTensor`: destination tensor (`tla::Tensor<LocalTensor, Layout, Coord, B2>`) in L0B

### l0Batch Overload (Ascend 950-specific)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t l0Batch);
```

- `l0Batch`: number of batches for batch data movement, which is used for consecutive movements across multiple batches in multi-batch scenarios.

### MX Scale Overload (Ascend 950-specific, B8/B4 zN → nZ/nZ → nZ)

```cpp
template <class TensorDst, class TensorSrc, class TensorMxScale>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorMxScale const &scaleTensor);
```

- `srcTensor`: source data tensor in L1. The element type is `float8_e4m3_t`, `float8_e5m2_t`, or `float4_*`, and the layout is zN (or nZ direct copy).
- `dstTensor`: destination tensor in L0B. The element type is `AscendC::mx_fp8_e4m3_t`, `AscendC::mx_fp8_e5m2_t`, or `float4_*`, and the layout is nZ.
- `scaleTensor`: MX scale tensor in L1. The element type is `float8_e8m0_t`, and the layout is nN (satisfying the `isMxScaleFornN` trait).

> **Note**: The B-side MX Scale uses an nN layout, which corresponds to the `isMxScaleFornN` trait. The scale data in Global Memory uses `tla::MakeMxScaleLayout<ElementMxScale, LayoutTag, true>(rows, cols)` (`isMxScaleB = true`) to create a layout, and then this layout is automatically converted into an nN layout after being transferred to L1 through TileCopyTla. `LayoutTagL1MxScaleB = layout::nN` is embedded in `PackedMxTileCopyTla` to manage B-side scale tensors.

## Examples

### zN → nZ Transposed Movement (Atlas A2, TLA)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// iszN<LayoutSrc> && isnZ<LayoutDst> → Automatic matching for transposed partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### nZ → nZ Direct Movement (Atlas A2, TLA, Transpose B)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// isnZ<LayoutSrc> && isnZ<LayoutDst> → Automatic matching for direct movement partial specialization
TileCopyTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### zN → nZ Transposed Movement (Ascend 950, TLA)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend950: zN L1 → nZ L0B
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### nZ → nZ Direct Movement (Ascend 950, TLA, Transpose B)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// Ascend 950: nZ L1 → nZ L0B (non-transposed direct movement)
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### l0Batch batch movement (Ascend 950, TLA)

```cpp
uint32_t l0Batch = 4;

// Ascend 950 supports l0Batch overload for continuous multi-batch data movement.
copyOp(dstTensor, srcTensor, l0Batch);
```

### MX Scale movement (Ascend 950, TLA, B-side)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float8_e4m3_t;
using ElementDst = AscendC::mx_fp8_e4m3_t;
using ElementMxScale = float8_e8m0_t;

const uint32_t K = 256;
const uint32_t N = 256;

const uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(K);

// Source data layout (L1 zN)
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(K, N);
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});

// Destination data layout (L0B nZ, element type: mx_fp8)
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout (L1 nN, isMxScaleB=true on the B side, constructed using MakeMxScaleLayout)
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

// MX Scale overload: L1 zN source data + L1 nN scale → L0B nZ mx data
TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```

### MX Scale movement (Ascend 950, TLA, B-side, nZ→nZ direct movement)

When the layout of the B-side data is nZ (in the Transpose B scenario), MX Scale is supported.

```cpp
// Both the source and destination data layouts are nZ.
auto layoutSrc = tla::MakeLayout<ElementSrc, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<ElementDst, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});

// MX Scale layout (L1 nN, isMxScaleB=true on the B side)
auto layoutScaleL1 = tla::MakeMxScaleLayout<ElementMxScale, layout::nN, true>(mxScaleK, N);

AscendC::LocalTensor<ElementMxScale> scaleL1Tensor;
auto scaleTensor = tla::MakeTensor(scaleL1Tensor, layoutScaleL1, Arch::PositionL1{});

TileCopyTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor, scaleTensor);
```
