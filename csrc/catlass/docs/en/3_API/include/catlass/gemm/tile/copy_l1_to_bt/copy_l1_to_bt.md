# CopyL1ToBT

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_bt.hpp)

[TOC]

## Function

`CopyL1ToBT` is a template that moves the Bias Table (a 1D vector) from L1 (Local Memory, referred to as A1 Buffer) to BT (Bias Table Buffer, referred to as C2 Buffer).

The Bias Table is used for bias addition and quantization/dequantization in matrix multiplication. Since bias data is a 1D vector, this template always uses `VectorLayout` (rank=1, stride=1) and performs continuous transfers via `AscendC::DataCopy` at `blockLen` granularity.

This template is not intended to be used directly in most cases. Instead, it is used as a member type of [TileCopy](../tile_copy/README.md) (`CopyL1ToBT`) and is automatically managed by `blockMmad`. Explicit declaration is only required during implementation of custom kernel templates that require manual assembly.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class L1Type,                     // L1 data description: Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>
    class L0Type = void               // BT data description: Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2>
>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};
```

- `ArchTag`: architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`.
- `L1Type`: data type of the bias table in L1. The value is fixed to `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>`.
- `L0Type`: data type of the bias table in the BT buffer. The value is fixed to `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2>`.

## Partial Specialization Implementation

| Architecture| Source Layout| Destination Layout| Location| Description|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| VectorLayout | VectorLayout | A1 → C2 | 1D vector copy using `AscendC::DataCopy`; block size based on `BYTE_PER_C2`|
| Ascend 950| VectorLayout | VectorLayout | A1 → C2 | 1D vector copy using `AscendC::DataCopy`; block size based on `BYTE_PER_C0`; automatic alignment for B32 data types|



## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> dstTensor,   // BT buffer destination tensor (C2)
    AscendC::LocalTensor<ElementSrc> srcTensor,   // L1 source tensor (A1)
    LayoutDst layoutDst,                          // BT data layout (VectorLayout)
    LayoutSrc layoutSrc                           // L1 data layout (VectorLayout)
);
```

- `srcTensor`: 1D bias table tensor in L1
- `dstTensor`: 1D bias table tensor in the C2 buffer (BT)
- The element type of `srcTensor` is `ElementSrc`, and the element type of `dstTensor` is `ElementDst`. The two types can be different (type conversion is supported).

## Examples

### Atlas A2

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### Ascend 950

```cpp
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = float;
using ElementDst = half;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstBTTensor;

using CopyOp = CopyL1ToBT<Arch::Ascend950, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstBTTensor, srcL1Tensor, layoutDst, layoutSrc);
```
