# TileElemwiseMul

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_mul.hpp)

[TOC]

## Function

`TileElemwiseMul` implements element-wise multiplication in the epilogue stage. It performs element-wise Mul on two input tensors in Unified Buffer and outputs to the destination tensor.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Computation completed using the `AscendC::Mul` instruction

## Template Prototype

```cpp
template <
    class ArchTag_,         // Architecture tag
    class ComputeType_,     // Computation data type (including Element)
    class TileShape_        // Tile shape type (including COUNT)
>
struct TileElemwiseMul;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `TileShape_` | Tile shape type. The total number of elements is obtained through `TileShape_::COUNT`.|

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

Internal implementation: `AscendC::Mul(ubOut, ubIn0, ubIn1, TileShape::COUNT)`

## Examples

```cpp
#include "catlass/epilogue/tile/tile_elemwise_mul.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using MulOp = TileElemwiseMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;

MulOp mulOp;
mulOp(ubOut, ubIn0, ubIn1);
```
