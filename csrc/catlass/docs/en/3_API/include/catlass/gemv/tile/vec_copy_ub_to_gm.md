# VecCopyUBToGm

> [Code Location](../../../../../../../include/catlass/gemv/tile/vec_copy_ub_to_gm.hpp)

[TOC]

## Function

`VecCopyUBToGm` implements vector data movement from Unified Buffer (UB) to Global Memory (GM) for GEMV scenarios. It uses `AscendC::DataCopyPad` to write a one-dimensional vector from UB back to GM.

- Applicability: Atlas A2
- `atomic_add` mode is supported. When `is_atoadd = true`, the template calls `AscendC::SetAtomicAdd<Element>()` before performing the data movement.

## Template Prototype

```cpp
template <class ArchTag, class GmType, bool is_atoadd = false>
struct VecCopyUBToGm;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `GmType` | `Gemm::GemmType<Element, VectorLayout>` |
| `is_atoadd` | Whether to enable the atomic_add mode.|

## Partial Specialization Implementation

| Architecture| GmType | is_atoadd | Description|
| :------ | :------ | :------ | :------ |
| Atlas A2| `VectorLayout` | `false` (default)| Standard movement through `DataCopyPad`|
| Atlas A2| `VectorLayout` | `true` | `SetAtomicAdd` + `DataCopyPad` |

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<Element> dstTensor,             // Destination GlobalTensor on Global Memory
    AscendC::LocalTensor<Element> srcTensor,              // Source LocalTensor on Unified Buffer (UB)
    layout::VectorLayout const &layoutDst,
    layout::VectorLayout const &layoutSrc
)
```

## Examples

### Default Mode

```cpp
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using GmType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t length = 64;
auto layoutSrc = layout::VectorLayout::MakeLayout<Element>(length, 1);
auto layoutDst = layout::VectorLayout::MakeLayout<Element>(length, 1);

AscendC::LocalTensor<Element> srcTensor;
AscendC::GlobalTensor<Element> dstTensor;

using CopyOp = VecCopyUBToGm<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```

### Atomic Add Mode

```cpp
using CopyOp = VecCopyUBToGm<Arch::AtlasA2, GmType, true>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
