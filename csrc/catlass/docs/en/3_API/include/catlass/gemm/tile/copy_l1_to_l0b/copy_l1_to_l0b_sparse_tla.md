# CopyL1ToL0BSparseTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp)

[TOC]

## Function

`CopyL1ToL0BSparseTla` is a sparse data movement template of the Tensor Layout Abstraction (TLA) style. This template is dedicated to L1 → L0B movements on the Atlas A2 architecture. In Sparse General Matrix Multiply (GEMM) scenarios, the B matrix is stored in a compressed format containing only non-zero elements. As a result, the data movement requires an index tensor to indicate which elements are valid.

Unlike the generic [TileCopyTla](./tile_copy_tla.md), `CopyL1ToL0BSparseTla` provides an `operator()` that accepts three tensor parameters: **src**, **dst**, and **index**. The sparse-indexed data movement is performed via the `AscendC::LoadDataWithSparse` instruction.

> **Restriction**: This template supports only the Atlas A2 architecture (CATLASS_ARCH == 2201) and is not supported on Ascend 950.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag: Arch::AtlasA2
    class ElementA,                   // B matrix element type
    class TensorSrc,                  // Source tensor: tla::Tensor<LocalTensor<Element>, Layout, Coord, A1>
    class TensorDst,                  // Destination tensor: tla::Tensor<LocalTensor<Element>, Layout, Coord, B2>
    class TensorIdx,                  // Index tensor: tla::Tensor<LocalTensor<uint8_t>, Layout, Coord, A1>
    class Enable = void               // SFINAE condition
>
struct CopyL1ToL0BSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0b sparse, can not find the specialization.");
};
```

- `ArchTag`: architecture tag. Only `Arch::AtlasA2` is supported.
- `ElementA`: element type of matrix B
- `TensorSrc`: source tensor in L1
- `TensorDst`: destination tensor in L0B
- `TensorIdx`: sparse index tensor in L1 (element type: `uint8_t`)

## Partial Specialization Implementation

### Atlas A2

| Source Tensor| Destination Tensor| Index Tensor | SFINAE Condition| Description|
| :------ | :------ | :------ | :------ | :------ |
| zN L1 | nZ L0B | zN L1 (`uint8_t`) | `iszN<LayoutSrc> && isnZ<LayoutDst> && iszN<LayoutIdx>` | Sparse transposed copy|
| nZ L1 | nZ L0B | nZ L1 (`uint8_t`) | `isnZ<LayoutSrc> && isnZ<LayoutDst> && isnZ<LayoutIdx>` | Sparse non-transposed copy (Transpose B)|

## APIs

```cpp
template <class TensorDst, class TensorSrc, class TensorIdx>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, TensorIdx const &idxTensor);
```

- `dstTensor`: destination tensor (`tla::Tensor<LocalTensor, Layout, Coord, B2>`) in L0B
- `srcTensor`: source tensor (`tla::Tensor<LocalTensor, Layout, Coord, A1>`) of the compressed matrix B in L1
- `idxTensor`: sparse index tensor in L1, with element type `uint8_t` and the same layout as srcTensor (`zN` or `nZ`).

In the index tensor, the upper 4 bits and lower 4 bits of each `uint8_t` element respectively store the index information for two adjacent elements. The indices are aligned via `INDEX_SHIFT = 2` for offset alignment.

## Examples

### zN → nZ Sparse Transposed Movement (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t K = 256;
const uint32_t N = 256;

// Layout of matrix B data (L1 zN)
auto layoutSrc = tla::MakeLayout<half, layout::zN>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);

// Sparse index layout (L1 zN, with element type uint8_t)
auto layoutIdx = tla::MakeLayout<uint8_t, layout::zN>(K, N);

AscendC::LocalTensor<half> srcL1Tensor;
AscendC::LocalTensor<half> dstL0BTensor;
AscendC::LocalTensor<uint8_t> idxL1Tensor;
auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});
auto idxTensor = tla::MakeTensor(idxL1Tensor, layoutIdx, Arch::PositionL1{});

// Instantiation and call
CopyL1ToL0BSparseTla<Arch::AtlasA2, half, decltype(srcTensor), decltype(dstTensor), decltype(idxTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor, idxTensor);
```

### nZ → nZ Sparse Non-Transposed Movement (Atlas A2, Transpose B)

```cpp
auto layoutSrc = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutDst = tla::MakeLayout<half, layout::nZ>(K, N);
auto layoutIdx = tla::MakeLayout<uint8_t, layout::nZ>(K, N);

auto srcTensor = tla::MakeTensor(srcL1Tensor, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstL0BTensor, layoutDst, Arch::PositionL0B{});
auto idxTensor = tla::MakeTensor(idxL1Tensor, layoutIdx, Arch::PositionL1{});

CopyL1ToL0BSparseTla<Arch::AtlasA2, half, decltype(srcTensor), decltype(dstTensor), decltype(idxTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor, idxTensor);
```
