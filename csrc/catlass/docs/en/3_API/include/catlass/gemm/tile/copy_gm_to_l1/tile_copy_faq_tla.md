# TileCopyFAQTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)

[TOC]

## Function

`TileCopyFAQTla` is a TLA-style FlashAttention LoadQ data movement template. It moves multi-matrix data from global memory to L1 and converts it into the zN fractal format, serving the Q-matrix preloading phase of FlashAttention. The source data is 3D (`ndNum` × `nValue` × `dValue`) and is converted to zN format in one step using `Nd2NzParams`.

> **Restriction**: This template supports only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) and is not supported on Ascend 950.

## Template Prototype

```cpp
template <
    class ArchTag,        // Architecture tag
    class TensorSrc,      // Source tensor (global memory, 3D multi-matrix)
    class TensorDst       // Destination tensor (L1, zN)
>
struct TileCopyFAQTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyFAQTla, can not find the specialization.");
};
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Only `Arch::AtlasA2` is supported.|
| `TensorSrc` | Source tensor, `tla::Tensor<GlobalTensor<Element>, Layout, Coord, GM>`, 3D shape|
| `TensorDst` | Destination tensor, `tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>`, in zN format|

## Partial Specialization Implementation

### Atlas A2

| Source Shape| Destination Layout| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| 3D (ndNum, n, d)| zN | `iszN<LayoutDst>` | Nd2Nz multi-matrix conversion with large-stride fallback to row-by-row processing|

- `ndNum`: number of matrices (corresponding to the sequence-length dimension)
- `nValue`: row dimension
- `dValue`: column dimension
- When `srcNdMatrixStride < STRIDE_LIMIT(65536)` is used, perform Nd2Nz movement in one go. Otherwise, perform matrix-by-matrix, row-by-row rollback.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`: destination tensor on L1 (zN format)
- `srcTensor`: source tensor on global memory (3D multi-matrix, for example, RowMajor 3D)

## Examples

### FlashAttention Q Matrix Loading (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t seqLen = 128;   // ndNum (number of matrices)
const uint32_t headDim = 64;   // dValue (head dimension)
const uint32_t numHeads = 32;  // nValue

// 3D source layout: RowMajor (seqLen, numHeads, headDim)
auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(seqLen, numHeads, headDim);
// L1 zN destination layout
auto dstLayout = tla::MakeLayout<half, layout::zN>(seqLen * numHeads, headDim);

AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopyFAQTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> faqCopyOp;
faqCopyOp(dstTensor, srcTensor);
```
