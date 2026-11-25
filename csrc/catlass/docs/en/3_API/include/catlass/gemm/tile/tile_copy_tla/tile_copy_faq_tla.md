# TileCopyFAQTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopyFAQTla` is a dedicated Tensor Layout Abstraction (TLA) data movement template for FlashAttention LoadQ. It performs a multi-matrix DataCopy from Global Memory (RowMajor layout) to L1 (zN layout). The DataCopy operation internally handles the layout conversion from ND to NZ.

Application scenario: FlashAttention Q matrix loading. Unlike the general-purpose `TileCopyTla`, TileCopyFAQTla fixes the conversion path to ND → NZ.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Primary Template Declaration

```cpp
template <
    class ArchTag,           // Architecture tag
    class TensorSrc,         // Source tensor (RowMajor GM)
    class TensorDst          // Destination tensor (zN L1)
>
struct TileCopyFAQTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyFAQTla, can not find the specialization.");
};
```

## Partial Specialization Implementation (Atlas A2)

| Direction| Implementation Location| API Reference|
| :------ | :------ | :------ |
| GM RowMajor → L1 zN | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_faq_tla.md) |

Difference from the generic `TileCopyTla` GM→L1 data movement: The DataCopy parameter uses `col * sizeof(Element)` instead of `col * sizeof(Element)/ELE_NUM_PER_BLK`. This means the copy is performed per-column rather than as 32B-aligned block copies.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // L1 zN tensor
    TensorSrc const &srcTensor     // GM RowMajor tensor
);
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

// FA LoadQ: GM RowMajor → L1 zN
auto qGmLayout = tla::MakeLayout<Element, layout::RowMajor>(seq_len, head_dim);
auto qL1Layout = tla::MakeLayout<Element, layout::zN>(seq_len, head_dim);
auto qGmTensor = tla::MakeTensor(qGm, qGmLayout, Arch::PositionGM{});
auto qL1Tensor = tla::MakeTensor(qL1, qL1Layout, Arch::PositionL1{});

TileCopyFAQTla<Arch::AtlasA2, decltype(qGmTensor), decltype(qL1Tensor)> copyOp;
copyOp(qL1Tensor, qGmTensor);
```
