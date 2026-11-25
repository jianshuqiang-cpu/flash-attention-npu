# TileBroadcastInplaceByColumn

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_broadcast_inplace_by_column.hpp)

[TOC]

## Function

`TileBroadcastInplaceByColumn` implements the in-place copy operation of column broadcasting in the epilogue stage. It broadcasts the first-column elements of a Unified Buffer tile (m,  n) to all other columns in the same row, overwriting in place. This is commonly used to expand a column vector (such as per-channel scale or zero-point) across the entire matrix.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Broadcasting using `AscendC::Copy` and `CopyRepeatParams`

## Template Prototype

```cpp
template <
    class ArchTag_,         // Architecture tag
    class ComputeType_,     // Computation data type (including Element)
    class TileShape_        // Tile shape type (including ROW and COLUMN)
>
struct TileBroadcastInplaceByColumn;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `TileShape_` | Tile shape. `TileShape_::ROW` indicates the number of rows, and `TileShape_::COLUMN` indicates the number of columns.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubInOut   // Unified Buffer input and output tensors (in-place)
)
```

| Parameter| Description|
| :------ | :------ |
| `ubInOut` | Unified Buffer tensor. The input is an (m, n) matrix, and the output has the first column broadcast to all columns.|

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_inplace_by_column.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastOp = TileBroadcastInplaceByColumn<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubInOut;

BroadcastOp broadcastOp;
broadcastOp(ubInOut);
```
