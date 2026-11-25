# TileCopySparseTla (GM → L1, Sparse)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_gm_to_l1.hpp)

[TOC]

## Function

`TileCopySparseTla` is a TLA-style template for Sparse GEMM data movement (GM → L1). In the Sparse GEMM scenario, matrix A is stored in compressed format (only non-zero elements). `TileCopySparseTla` handles moving the sparse matrix A from global memory to L1 and converts the source data, whether row-major or column-major, into zN or nZ fractal format.

> **Restriction**: This template supports only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) and is not supported on Ascend 950.

## Template Prototype

```cpp
template <
    class ArchTag,        // Architecture tag
    class TensorSrc,      // Source tensor (global memory)
    class TensorDst,      // Destination tensor (L1)
    class Enable = void   // SFINAE distribution
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Only `Arch::AtlasA2` is supported.|
| `TensorSrc` | Source tensor, `tla::Tensor<GlobalTensor<Element>, Layout, Coord, GM>`|
| `TensorDst` | Destination tensor, `tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>`|
| `Enable` | SFINAE condition, which automatically dispatches partial specialization based on layout|

## Partial Specialization Implementation

### Atlas A2

| Source Layout| Destination Layout| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ |
| RowMajor/ColumnMajor| zN | `(isRowMajor \|\| isColumnMajor) && iszN` | ND → zN conversion through `Nd2NzParams`|
| ColumnMajor | nZ | `isColumnMajor && isnZ` | ND → nZ conversion, with large-stride row-by-row rollback|
| zN (uint32_t compression)| zN | `iszN<uint32_t> && iszN` | zN → zN direct transfer, with 16-byte alignment handling|
| nZ (uint32_t compression)| nZ | `isnZ<uint32_t> && isnZ` | nZ → nZ direct transfer|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`: destination tensor on L1 (zN or nZ format)
- `srcTensor`: source tensor on global memory (RowMajor/ColumnMajor/zN/nZ)

## Examples

### RowMajor GM → zN L1 (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t K = 256;

auto srcLayout = tla::MakeLayout<half, layout::RowMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::zN>(M, K);

AscendC::GlobalTensor<half> srcGmTensor;
AscendC::LocalTensor<half> dstL1Tensor;

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```

### ColumnMajor GM → nZ L1 (Atlas A2)

```cpp
auto srcLayout = tla::MakeLayout<half, layout::ColumnMajor>(M, K);
auto dstLayout = tla::MakeLayout<half, layout::nZ>(M, K);

auto srcTensor = tla::MakeTensor(srcGmTensor, srcLayout, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstL1Tensor, dstLayout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
