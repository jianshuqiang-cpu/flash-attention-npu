# TileCopySparseTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopySparseTla` is a Tensor Layout Abstraction (TLA) movement template dedicated to sparse General Matrix Multiply (GEMM). The core difference from [TileCopyTla](./tile_copy_tla.md) is that the source and destination tensors may be of different element types (for example, the B index data is `int32_t`). The structure is similar to that of `TileCopyTla` and is automatically dispatched through SFINAE.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Primary Template Declaration

```cpp
template <
    class ArchTag,           // Architecture tag
    class TensorSrc,         // Source tensor type
    class TensorDst,         // Destination tensor type
    class Enable = void      // SFINAE enable
>
struct TileCopySparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopySparseTla, can not find the specialization.");
};
```

## Partial Specialization Implementation (Atlas A2)

| Direction| Description| Implementation Location| API Reference|
| :------ | :------ | :------ | :------ |
| GM→L1A | Sparse matrix A: GM RowMajor → L1 RowMajor| `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_sparse_tla.md) |
| GM→L1B | Sparse matrix B: GM ColumnMajor → L1 ColumnMajor| `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_sparse_tla.md) |
| L1→L0A | Sparse matrix A: L1 → L0A (zZ layout)| `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_sparse_tla.md) |

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor
);
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

// Matrix B index (int32_t) GM → L1
auto idxGmLayout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K, N);
auto idxL1Layout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K, N);
auto idxGmTensor = tla::MakeTensor(idxGm, idxGmLayout, Arch::PositionGM{});
auto idxL1Tensor = tla::MakeTensor(idxL1, idxL1Layout, Arch::PositionL1{});

TileCopySparseTla<Arch::AtlasA2, decltype(idxGmTensor), decltype(idxL1Tensor)> copyOp;
copyOp(idxL1Tensor, idxGmTensor);
```
