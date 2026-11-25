# CopyL1ToFP

> [Code Location](../../../../../../../include/catlass/gemm/tile/copy_l1_to_fp.hpp)

[TOC]

## Function

`CopyL1ToFP` is a template that moves data (typically auxiliary information such as quantization Scale) from L1 (Local Memory, referred to as A1 Buffer) to FP (FixPipe Buffer, referred to as C2PIPE2GM). This movement writes the data directly toGlobal Memory (GM) via the FixPipe channel.

FixPipe is a special data path in the Atlas A2 architecture. It allows the direct movement of computation results or auxiliary data from the core to external storage (GM), bypassing the conventional L0C → GM path. It is commonly used for Scale data write-back in per-token/per-channel dequantization scenarios.

This template is not intended to be used directly in most cases. Instead, it is used as a member type of [TileCopy](./tile_copy/README.md) (`CopyL1ToFP`) and is automatically managed by `blockMmad`. Explicit declaration is only required during implementation of custom kernel templates that require manual assembly.

> **Restriction**: This template supports only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) and is not supported on Ascend 950.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag: Arch::AtlasA2
    class L1Type,                     // L1 data description: Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>
    class L0Type = void               // FP data description: Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>
>
struct CopyL1ToFP {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to fixpipe buffer, can not find the specialization.");
};
```

- `ArchTag`: architecture tag. Only `Arch::AtlasA2` is supported.
- `L1Type`: 1D vector data type in L1. The value is fixed to `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>`.
- `L0Type`: data type of the FixPipe buffer. The value is fixed to `Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>`.

## Partial Specialization Implementation

| Architecture| Source Layout| Destination Layout| Location| Description|
| :------ | :------ | :------ | :------ | :------ |
| Atlas A2| VectorLayout | VectorLayout | A1 → C2PIPE2GM | 1D vector copy using `AscendC::DataCopy`; block size based on `BYTE_PER_BLK_FP`|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> dstTensor,   // FixPipe destination tensor (C2PIPE2GM)
    AscendC::LocalTensor<ElementSrc> srcTensor,   // L1 source tensor (A1)
    LayoutDst layoutDst,                          // FixPipe data layout (VectorLayout)
    LayoutSrc layoutSrc                           // L1 data layout (VectorLayout)
);
```

- `srcTensor`: 1D source tensor in L1
- `dstTensor`: 1D destination tensor in the FixPipe buffer
- The element type of `srcTensor` is `ElementSrc`, and that of `dstTensor` is `ElementDst`. The two types can be different.

## Examples

### Basic FixPipe-based Movement (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l1_to_fp.hpp"

using namespace Catlass::Gemm::Tile;

using ElementSrc = uint64_t;
using ElementDst = uint64_t;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>;

uint32_t vecLen = 256;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstFPTensor;

using CopyOp = CopyL1ToFP<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstFPTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### FixPipe-based Type Conversion Movement

The source and destination element types can be different. Type conversion is supported during movement.

```cpp
using ElementSrc = float;
using ElementDst = uint64_t;
using L1Type = Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>;
using L0Type = Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>;

auto layoutSrc = layout::VectorLayout(vecLen);
auto layoutDst = layout::VectorLayout(vecLen);

AscendC::LocalTensor<ElementSrc> srcL1Tensor;
AscendC::LocalTensor<ElementDst> dstFPTensor;

using CopyOp = CopyL1ToFP<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstFPTensor, srcL1Tensor, layoutDst, layoutSrc);
```
