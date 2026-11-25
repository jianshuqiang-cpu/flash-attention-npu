# TileRowBroadcastAdd

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_broadcast_add.hpp)

[TOC]

## Function

`TileRowBroadcastAdd` implements the row-broadcast addition operation in the epilogue stage. It takes a (1,  n) row vector, broadcasts it to (m, n), and performs element-wise addition with another (m, n) matrix, outputting the result to the destination tensor.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Broadcast addition implemented via `AscendC::Add` and `BinaryRepeatParams`

## Template Prototype

```cpp
template <
    class ArchTag_,       // Architecture tag
    class ComputeType_,   // Computation data type (including Element)
    class TileShape_      // Tile shape type (including ROW and COLUMN)
>
struct TileRowBroadcastAdd;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `TileShape_` | Tile shape. `TileShape_::COLUMN` is used to compute the block division parameters.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,           // Destination Unified Buffer LocalTensor
    AscendC::LocalTensor<ElementCompute> const &ubIn0,           // Source Unified Buffer LocalTensor 0 (m, n)
    AscendC::LocalTensor<ElementCompute> const &ubIn1,           // Source Unified Buffer LocalTensor 1 (1, n) row vector
    MatrixCoord const &actualTileShape                           // Actual tile shape (m, n)
)
```

| Parameter| Description|
| :------ | :------ |
| `ubOut` | Destination Unified Buffer tensor, which stores the `ubIn0[i] + ubIn1` result|
| `ubIn0` | Unified Buffer tensor of shape (m, n)|
| `ubIn1` | Row vector of shape (1, n), which is added element-wise to ubIn0 after broadcasting|
| `actualTileShape` | `MatrixCoord{rows, cols}`, actual tile dimension|

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_add.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastAddOp = TileRowBroadcastAdd<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;
MatrixCoord actualTileShape(128, 256);

BroadcastAddOp broadcastAddOp;
broadcastAddOp(ubOut, ubIn0, ubIn1, actualTileShape);
```
