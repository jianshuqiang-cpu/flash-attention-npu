# CopyL1ToL0B

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l1_to_l0b.hpp)

[TOC]

## Function

`CopyL1ToL0B` is a template responsible for moving tile blocks of matrix B from L1 (Local Memory, also referred to as the B1 Buffer) to L0B (the B2 Buffer). It supports conversion between multiple data layouts.

Depending on the source and destination layouts, the template selects an appropriate hardware data movement instruction.
- **zZ → nZ**: transposed copy (Transpose B) with `ifTranspose = true`. For float data, the `LoadDataWithTranspose` instruction is used.
- **zN → nZ**: transposed copy. For int8_t data, the `LoadDataWithTranspose` instruction is used. For float data, the `LoadData3D and SetFmatrix` instructions are used.
- **nZ → nZ**: non-transposed copy (direct copy), with `ifTranspose = false`
- **zN → zN/nN → zN**: GEMV-specific path

This template is not intended to be used directly in most cases. Instead, it is used as a member type of [TileCopy](../tile_copy/README.md) and is automatically managed by `blockMmad`. Explicit declaration is only required during implementation of custom kernels that require manual assembly.

## Template Prototype

```cpp
template <
    class ArchTag,                    // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class L1Type,                     // L1 data description: Gemm::GemmType<Element, Layout, AscendC::TPosition::B1>
    class L0Type = void               // L0B data description: Gemm::GemmType<Element, Layout, AscendC::TPosition::B2> (optional, automatically deduced)
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
```

- `ArchTag`: architecture tag, which can be `Arch::AtlasA2` or `Arch::Ascend950`
- `L1Type`: data type of matrix B in L1, which encapsulates Element, Layout, and TPosition.
- `L0Type`: data type of matrix B in L0B. The default value is `void`. Most partial specializations will automatically deduce it.

## Partial Specialization Implementation

### Partial Specializations (Atlas A2 — GEMM scenarios: zZ/zN → nZ)

| Source Layout| Destination Layout| Element Type| Description|
| :------ | :------ | :------ | :------ |
| zZ (B1)| nZ (B2)| Any| Basic transposed copy using LoadData2D (ifTranspose=true)|
| zZ (B1)| nZ (B2)| float | float-specific path using LoadData2dTranspose|
| zN (B1)| nZ (B2)| int8_t | int8_t transposed copy using LoadDataWithTranspose|
| zN (A1)| nZ (B2)| int8_t | int8_t zN → nZ transposed copy (single-parameter L1Type specialization)|
| zN (A1)| nZ (B2)| float | float zN → nZ transposed copy using LoadData3D + SetFmatrix|
| zN (A1)| nZ (B2)| Any (non-int8_t/float)| Generic zN → nZ transposed copy|
| zN (A1)| nZ (B2)| AscendC::int4b_t | int4b_t zN → nZ transposed copy using LoadDataWithTranspose|
| nZ (B1)| nZ (B2)| Any| nZ → nZ non-transposed copy (direct copy)|
| nZ (A1)| nZ (B2)| Any| nZ → nZ direct copy (single-parameter L1Type specialization)|

### Partial Specializations (Atlas A2 — GEMV scenarios: zN/nN → zN)

| Source Layout| Destination Layout| Element Type| Description|
| :------ | :------ | :------ | :------ |
| zN (B1)| zN (B2)| Any| GEMV-specific zN → zN copy|
| nN (B1)| zN (B2)| Any| GEMV-specific nN → zN transposed copy|
| nN (B1)| zN (B2)| float | float-specific GEMV nN → zN transposed copy|
| nZ (B1)| zN (B2)| int8_t | int8_t GEMV nZ → zN transposed copy|

### Ascend 950

| Source Layout| Destination Layout| Element Type| Description|
| :------ | :------ | :------ | :------ |
| nZ (A1)| nZ (B2)| Any| nZ → nZ non-transposed copy using LoadData2DParamsV2|
| zN (A1), non-B8/B4| nZ (B2)| Not int8_t/float8_/float4| zN → nZ transposed copy using LoadData2DParamsV2|
| zN (A1), B8/B4| nZ (B2)| int8_t/float8_/float4| B8/B4 zN → nZ transposed copy. Select single-step or step-by-step LoadData based on the L0N alignment.|

> **Note**: The destination layout of L0Type on Ascend 950 is nZ (not zZ), and the position of L1Type is A1 (not B1), which are different from those on Atlas A2.

## APIs

### Basic APIs (For All Partial Specializations)

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,   // L0B destination tensor.
    AscendC::LocalTensor<Element> srcTensor,   // L1 Source tensor
    LayoutDst layoutDst,                       // L0B data layout
    LayoutSrc layoutSrc                        // L1 data layout
);
```

## Examples

### zZ → nZ Transposed Movement (Atlas A2, non-TLA)

```cpp
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

using namespace Catlass::Gemm::Tile;

using Element = half;
using L1Type = Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>;

uint32_t k = 256;
uint32_t n = 256;

auto layoutSrc = layout::zZ::MakeLayout<Element>(k, n);
auto layoutDst = layout::nZ::MakeLayout<Element>(k, n);

AscendC::LocalTensor<Element> srcL1Tensor;
AscendC::LocalTensor<Element> dstL0BTensor;

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### zN → nZ Transposed Movement (Atlas A2, Single-Parameter L1Type)

```cpp
using L1Type = Gemm::GemmType<half, layout::zN, AscendC::TPosition::A1>;

auto layoutSrc = layout::zN::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

// Single-parameter L1Type specialization, with L0Type automatically deduced
using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### nZ → nZ Direct Copy (Atlas A2, non-TLA)

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::B1>;
using L0Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::B2>;

auto layoutSrc = layout::nZ::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

using CopyOp = CopyL1ToL0B<Arch::AtlasA2, L1Type, L0Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```

### nZ → nZ Movement (Ascend 950, non-TLA)

```cpp
using L1Type = Gemm::GemmType<half, layout::nZ, AscendC::TPosition::A1>;

auto layoutSrc = layout::nZ::MakeLayout<half>(k, n);
auto layoutDst = layout::nZ::MakeLayout<half>(k, n);

// For Ascend 950, L1Type Position is A1, and L0Type is optional.
using CopyOp = CopyL1ToL0B<Arch::Ascend950, L1Type>;
CopyOp copyOp;
copyOp(dstL0BTensor, srcL1Tensor, layoutDst, layoutSrc);
```
