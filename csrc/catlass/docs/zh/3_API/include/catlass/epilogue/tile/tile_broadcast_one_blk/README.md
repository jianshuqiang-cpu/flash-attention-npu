# tile_broadcast_one_blk

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## 概述

`tile_broadcast_one_blk` 模块实现 epilogue 阶段的 one-block 广播操作。将 UB 上的单个元素广播到整个 block（32B），常用于将 scalar scale/zero 点广播后参与向量计算。

## API 清单

| API                                                       | 风格   | 说明                                             |
| :-------------------------------------------------------- | :----- | :----------------------------------------------- |
| [TileBroadcastOneBlk](./tile_broadcast_one_blk.md)        | 非 TLA | `AscendC::Brcb` + `BrcbRepeatParams`             |
| [TileBroadcastOneBlkTla](./tile_broadcast_one_blk_tla.md) | TLA    | TLA 版本，`tensor.layout()(tensor.coord())` 偏移 |

## 调用示例

### TileBroadcastOneBlk（非 TLA）

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

### TileBroadcastOneBlkTla（TLA）

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
