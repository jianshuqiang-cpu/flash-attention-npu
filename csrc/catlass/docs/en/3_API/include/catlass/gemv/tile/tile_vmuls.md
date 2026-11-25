# TileVmuls

> [Code Location](../../../../../../../include/catlass/gemv/tile/tile_vmuls.hpp)

[TOC]

## Function

`TileVmuls` is a template that implements vector-scalar multiplication for GEMV scenarios. It applies `AscendC::Muls` to perform element-wise multiplication of a vector (residing in UB) by a scalar value.

- It applies to all architectures (no specialization).
- The effective processing length is managed by `AscendC::SetVectorMask<Element, MaskMode::COUNTER>(len)`.

## Template Prototype

```cpp
template <class ArchTag, class VType_>
struct TileVmuls;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `VType_` | Vector data type. The element type is obtained through `VType_::Element`.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // Destination UB LocalTensor
    AscendC::LocalTensor<Element> srcTensor,    // Source UB LocalTensor
    Element scalar,                             // Scalar value
    uint32_t len                                // Vector length
)
```

## Examples

```cpp
#include "catlass/gemv/tile/tile_vmuls.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using VType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t len = 64;
Element scale = 0.5f;

AscendC::LocalTensor<Element> dstTensor, srcTensor;

using VmulsOp = TileVmuls<Arch::AtlasA2, VType>;
VmulsOp vmuls;
vmuls(dstTensor, srcTensor, scale, len);
```
