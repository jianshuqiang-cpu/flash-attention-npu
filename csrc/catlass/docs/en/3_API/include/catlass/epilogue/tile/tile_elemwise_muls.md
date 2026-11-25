# TileElemWiseMuls

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_muls.hpp)

[TOC]

## Function

`TileElemWiseMuls` implements element-wise scalar multiplication in the epilogue stage. It multiplies each element of the input tensor in Unified Buffer by a scalar value and outputs the result.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Computation completed using the `AscendC::Muls` instruction

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ComputeType_,       // Computation data type (including Element)
    uint32_t COMPUTE_LENGTH_  // Computation length (number of elements)
>
struct TileElemWiseMuls;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `COMPUTE_LENGTH_` | Computation length|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> dstLocal,     // Destination Unified Buffer LocalTensor
    AscendC::LocalTensor<ElementCompute> srcTensor,    // Source Unified Buffer LocalTensor
    ElementCompute scalar                              // Scalar value
)
```

| Parameter| Description|
| :------ | :------ |
| `dstLocal` | Destination Unified Buffer tensor (overwritten), which stores the `src[i] * scalar` result|
| `srcTensor` | Source Unified Buffer tensor|
| `scalar` | Scalar multiplier|

Internal implementation: `AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH)`

## Examples

```cpp
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using MulsOp = TileElemWiseMuls<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstTensor;
AscendC::LocalTensor<half> srcTensor;
half scalar = 0.5_h;

MulsOp mulsOp;
mulsOp(dstTensor, srcTensor, scalar);
```
