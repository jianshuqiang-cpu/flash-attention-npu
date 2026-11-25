# CopyL1ToL0BSparseTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`CopyL1ToL0BSparseTla` is a Tile-Level Abstraction (TLA) template for moving the B matrix from L1 to L0B in sparse General Matrix Multiplication (GEMM). It requires an additional `TensorIdx` (a sparse index tensor of type `int32_t`) to assist in sparse decompression. Unlike the regular [CopyL1ToL0B](../copy_l1_to_l0b/tile_copy_tla.md), it uses the index to decompress the dense B matrix into corresponding blocks during the transfer to L0B.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Primary Template Declaration

```cpp
template <
    class ArchTag,            // Architecture tag
    class Element,            // A matrix element type (affecting L0B column)
    class TensorSrc,          // B matrix L1 tensor
    class TensorDst,          // B matrix L0B tensor
    class TensorIdx,          // Sparse index tensor
    class Enable = void       // SFINAE enable
>
struct CopyL1ToL0BSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported CopyL1ToL0BSparseTla, can not find the specialization.");
};
```

## Partial Specialization Implementation (Atlas A2)

| Condition| Description| Implementation Location| API Reference|
| :------ | :------ | :------ | :------ |
| `isSparseEnalbd` | B matrix ColumnMajor→nZ + index | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/copy_l1_to_l0b_sparse_tla.md) |

## APIs

```cpp
template <class TensorDst, class TensorSrc, class TensorIdx>
void operator()(
    TensorDst const &l0BTensor,    // L0B destination
    TensorSrc const &l1BTensor,    // L1 B source
    TensorIdx const &l1BIdxTensor  // L1 B index
);
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementA = half;
using ElementB = half;

// Transfer B matrix from L1 (column-major) to L0B (non-zero blocks)
auto l1bLayout = tla::MakeLayout<ElementB, layout::ColumnMajor>(K1, N1);
auto l0bLayout = tla::MakeLayout<ElementB, layout::nZ>(K1, N1);
auto l1bTensor = tla::MakeTensor(l1bData, l1bLayout, Arch::PositionL1{});
auto l0bTensor = tla::MakeTensor(l0bData, l0bLayout, Arch::PositionL0B{});

// Index tensor (int32_t, same L1 layout as B)
auto idxLayout = tla::MakeLayout<int32_t, layout::ColumnMajor>(K1, N1);
auto idxTensor = tla::MakeTensor(idxData, idxLayout, Arch::PositionL1{});

CopyL1ToL0BSparseTla<Arch::AtlasA2, ElementA,
    decltype(l1bTensor), decltype(l0bTensor), decltype(idxTensor)> copyOp;
copyOp(l0bTensor, l1bTensor, idxTensor);
```
