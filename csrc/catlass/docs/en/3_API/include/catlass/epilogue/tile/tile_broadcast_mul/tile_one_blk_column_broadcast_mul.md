# TileOneBlkColumnBroadcastMul

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## Function

`TileOneBlkColumnBroadcastMul` implements the column broadcast multiplication operation in the epilogue stage. A column vector (m,  1) is broadcast within a block to (m,  n) and then multiplied with the input. The broadcast granularity is a block (`BYTE_PER_BLK` bytes), meaning each single element along the column is broadcast to one full block.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA

## Template Prototype

```cpp
template <
    class ArchTag_,       // Architecture tag
    class ComputeType_,   // Computation data type
    class TileShape_      // Tile shape
>
struct TileOneBlkColumnBroadcastMul;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | `Gemm::GemmType<ElementCompute, RowMajor>` |
| `TileShape_` | Tile shape, `Shape<ROW, COLUMN>`|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,
    AscendC::LocalTensor<ElementCompute> const &ubIn0,
    AscendC::LocalTensor<ElementCompute> const &ubIn1     // (m, eleNumPerBlk) shape
)
```

Column broadcasting can be implemented using `AscendC::Mul` and `BinaryRepeatParams` (with `src1RepStride = 0` and `src1BlkStride = 1`).

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using ColumnBroadcastMul = TileOneBlkColumnBroadcastMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut, ubIn0, ubIn1;

ColumnBroadcastMul op;
op(ubOut, ubIn0, ubIn1);
```
