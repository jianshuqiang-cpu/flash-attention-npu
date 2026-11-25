# tile_broadcast_mul

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## Overview

The `tile_broadcast_mul` module implements the broadcast multiplication in the epilogue stage, supporting two broadcast modes: row broadcast and column broadcast (OneBlk). Each mode has both non-TLA and TLA implementations.

## API List

| API | Style| Broadcast Mode| Description|
| :------ | :------ | :------ | :------ |
| [TileRowBroadcastMul](./tile_row_broadcast_mul.md) | Non-TLA| Row broadcast| (1,n) → (m,n), `Mul` + `src1RepStride=0`|
| [TileRowBroadcastMulTla](./tile_row_broadcast_mul_tla.md) | TLA| Row broadcast| TLA version row broadcast multiplication|
| [TileOneBlkColumnBroadcastMul](./tile_one_blk_column_broadcast_mul.md) | Non-TLA| Column broadcast (OneBlk)| (m,1) → (m,n), `Mul` + block-level repetition|
| [TileOneBlkColumnBroadcastMulTla](./tile_one_blk_column_broadcast_mul_tla.md) | TLA | Column broadcast (OneBlk)| TLA version column broadcast multiplication|

## Examples

### TileRowBroadcastMul (non-TLA)

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

### TileRowBroadcastMulTla (TLA)

```cpp
constexpr uint32_t M = 128, N = 256;

auto layout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<half> ubOutData, ubIn0Data, ubIn1Data;
auto ubOut = tla::MakeTensor(ubOutData, layout, Arch::PositionUB{});
auto ubIn0 = tla::MakeTensor(ubIn0Data, layout, Arch::PositionUB{});
auto ubIn1 = tla::MakeTensor(ubIn1Data, layout, Arch::PositionUB{});

TileRowBroadcastMulTla<Arch::AtlasA2, half, Shape<M, N>> op;
op(ubOut, ubIn0, ubIn1);
```
