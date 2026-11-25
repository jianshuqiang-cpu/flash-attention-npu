# TileElemWiseAdd

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_add.hpp)

[TOC]

## Function

`TileElemWiseAdd` implements element-wise addition in the epilogue stage. It performs element-wise Add on two input tensors in Unified Buffer and outputs to the destination tensor.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Computation completed using the `AscendC::Add` instruction

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ComputeType_,       // Computation data type (including Element)
    uint32_t COMPUTE_LENGTH_  // Computation length (number of elements)
>
struct TileElemWiseAdd;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `COMPUTE_LENGTH_` | Computation length, which means the number of elements to be computed|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,   // Destination Unified Buffer LocalTensor
    AscendC::LocalTensor<ElementCompute> const &ubIn0,   // Source Unified Buffer LocalTensor 0
    AscendC::LocalTensor<ElementCompute> const &ubIn1    // Source Unified Buffer LocalTensor 1
)
```

| Parameter| Description|
| :------ | :------ |
| `ubOut` | Destination Unified Buffer tensor|
| `ubIn0` | First source Unified Buffer tensor|
| `ubIn1` | Second source Unified Buffer tensor|

Internal implementation: `AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH)`

## Examples

```cpp
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using AddOp = TileElemWiseAdd<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;

AddOp addOp;
addOp(ubOut, ubIn0, ubIn1);
```
