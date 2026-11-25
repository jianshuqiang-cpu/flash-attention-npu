# VecCopyGmToUB

> [Code Location](../../../../../../../include/catlass/gemv/tile/vec_copy_gm_to_ub.hpp)

[TOC]

## Function

`VecCopyGmToUB` implements vector data movement from Global Memory (GM) to Unified Buffer (UB) for GEMV (matrix-vector multiplication) scenarios. It uses `AscendC::DataCopy` to copy a one-dimensional vector of length `len` from GM to UB.

- It applies to all architectures (no specialization).
- Single-block movement (`blockCount = 1`), zero stride

## Template Prototype

```cpp
template <class ArchTag_, class VType_>
struct VecCopyGmToUB;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `VType_` | Vector data type. The element type is obtained through `VType_::Element`.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,      // Destination LocalTensor on the UB
    AscendC::GlobalTensor<Element> srcTensor,     // Source GlobalTensor on the GM
    uint32_t len                                   // Length of the moved element
)
```

## Examples

```cpp
#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using VType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t len = 64;

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = VecCopyGmToUB<Arch::AtlasA2, VType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, len);
```
