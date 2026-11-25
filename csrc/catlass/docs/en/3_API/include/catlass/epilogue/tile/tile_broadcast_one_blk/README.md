# tile_broadcast_one_blk

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## Overview

The `tile_broadcast_one_blk` module implements one-block broadcast in the epilogue stage. It broadcasts a single element from Unified Buffer across an entire block (32 bytes), which is commonly used to broadcast scalar scale or zero-point values so they can participate in vector computations.

## API List

| API | Style| Description|
| :------ | :------ | :------ |
| [TileBroadcastOneBlk](./tile_broadcast_one_blk.md) | Non-TLA| `AscendC::Brcb` + `BrcbRepeatParams` |
| [TileBroadcastOneBlkTla](./tile_broadcast_one_blk_tla.md) | TLA| TLA version, `tensor.layout()(tensor.coord())` offset|

## Examples

### TileBroadcastOneBlk (non-TLA)

```cpp
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 256;

using BroadcastOp = TileBroadcastOneBlk<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> ubOut, ubIn;
BroadcastOp broadcastOp;
broadcastOp(ubOut, ubIn);
```

### TileBroadcastOneBlkTla (TLA)

```cpp
constexpr uint32_t COMPUTE_LENGTH = 256;

auto layoutOut = tla::MakeLayout<half, layout::RowMajor>(COMPUTE_LENGTH, 32);
auto layoutIn = tla::MakeLayout<half, layout::VectorLayout>(COMPUTE_LENGTH, 1);

AscendC::LocalTensor<half> ubOutData, ubInData;
auto ubOut = tla::MakeTensor(ubOutData, layoutOut, Arch::PositionUB{});
auto ubIn = tla::MakeTensor(ubInData, layoutIn, Arch::PositionUB{});

TileBroadcastOneBlkTla<Arch::AtlasA2, half, COMPUTE_LENGTH> op;
op(ubOut, ubIn);
```
