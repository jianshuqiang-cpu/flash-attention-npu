# TileElemWiseGelu

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_gelu.hpp)

[TOC]

## Function

`TileElemWiseGelu` implements the Gaussian Error Linear Unit (GELU) activation function in the epilogue stage. It performs element-wise GELU computation on the input tensor in Unified Buffer and outputs the result.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- GELU approximation formula: `x / (1 + e^(-1.5957691 * 0.044715 * (x/0.044715 + x^3)))`
- Implementation: completed by combining the `Mul`, `Axpy`, `Muls`, `Exp`, `Adds`, and `Div` instructions

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ComputeType_,       // Computation data type (including Element)
    uint32_t COMPUTE_LENGTH_  // Computation length (number of elements)
>
struct TileElemWiseGelu;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `COMPUTE_LENGTH_` | Computation length|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &dstLocal,  // Destination Unified Buffer LocalTensor
    AscendC::LocalTensor<ElementCompute> const &srcLocal   // Source Unified Buffer LocalTensor
)
```

| Parameter| Description|
| :------ | :------ |
| `dstLocal` | Destination Unified Buffer tensor (overwritten), which stores the GELU(x) result|
| `srcLocal` | Source Unified Buffer tensor, with the input value x|

## Examples

```cpp
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using GeluOp = TileElemWiseGelu<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> srcLocal;

GeluOp geluOp;
geluOp(dstLocal, srcLocal);
```
