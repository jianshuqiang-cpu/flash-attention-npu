# tile_broadcast_mul

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## 概述

`tile_broadcast_mul` 模块实现 epilogue 阶段的广播乘法操作，包含行广播和列广播（OneBlk）两种广播模式，每种模式均有非 TLA 和 TLA 两套实现。

## API 清单

| API                                                                           | 风格   | 广播模式       | 说明                                   |
| :---------------------------------------------------------------------------- | :----- | :------------- | :------------------------------------- |
| [TileRowBroadcastMul](./tile_row_broadcast_mul.md)                            | 非 TLA | 行广播         | (1,n)→(m,n)，`Mul` + `src1RepStride=0` |
| [TileRowBroadcastMulTla](./tile_row_broadcast_mul_tla.md)                     | TLA    | 行广播         | TLA 版本行广播乘法                     |
| [TileOneBlkColumnBroadcastMul](./tile_one_blk_column_broadcast_mul.md)        | 非 TLA | 列广播（1Blk） | (m,1)→(m,n)，`Mul` + block 级重复      |
| [TileOneBlkColumnBroadcastMulTla](./tile_one_blk_column_broadcast_mul_tla.md) | TLA    | 列广播（1Blk） | TLA 版本列广播乘法                     |

## 调用示例

### TileRowBroadcastMul（非 TLA）

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

### TileRowBroadcastMulTla（TLA）

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
