# TileRowBroadcastMul

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## Function

`TileRowBroadcastMul` implements the broadcast multiplication operation in the epilogue stage. A row vector (1,  n) in Unified Buffer is broadcast to an (m,  n) matrix and then multiplied element-wise with the input. Row broadcasting can be implemented using `AscendC::Mul` and `BinaryRepeatParams` (`src1RepStride = 0`).

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA

## Template Prototype

```cpp
template <
    class ArchTag_,       // Architecture tag
    class ComputeType_,     // Computation data type
    class TileShape_      // Tile shape (including ROW and COLUMN)
>
struct TileRowBroadcastMul;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | `Gemm::GemmType<ElementCompute, RowMajor>` |
| `TileShape_` | Tile shape, `Shape<ROW, COLUMN>`|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,    // Destination Unified Buffer
    AscendC::LocalTensor<ElementCompute> const &ubIn0,    // Source Unified Buffer 0 (m, n)
    AscendC::LocalTensor<ElementCompute> const &ubIn1     // Source Unified Buffer 1 (1, n) row vector
)
```

Implement row broadcasting using `AscendC::Mul` and `BinaryRepeatParams` (`src1RepStride = 0`).

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;
using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastMul = TileRowBroadcastMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut, ubIn0, ubIn1;

BroadcastMul broadcastMul;
broadcastMul(ubOut, ubIn0, ubIn1);
```
