# TileElemWiseSilu

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_silu.hpp)

[TOC]

## Function

`TileElemWiseSilu` implements the Sigmoid Linear Unit (SiLU) activation function, also known as the Swish function, in the epilogue stage. It performs element-wise computation of `x * sigmoid(x)` on the input tensor in Unified Buffer and outputs the result.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Formula: `SiLU(x) = x/(1 + e^(-x))`
- Implementation: completed by combining the `Muls` (negation), `Exp`, `Adds`, and `Div` instructions

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ComputeType_,       // Computation data type (including Element)
    uint32_t COMPUTE_LENGTH_  // Computation length (number of elements)
>
struct TileElemWiseSilu;
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
| `dstLocal` | Destination Unified Buffer tensor (overwritten), which stores the SiLU(x) result|
| `srcLocal` | Source Unified Buffer tensor, with the input value x|

## Examples

```cpp
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using SiluOp = TileElemWiseSilu<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> srcLocal;

SiluOp siluOp;
siluOp(dstLocal, srcLocal);
```
