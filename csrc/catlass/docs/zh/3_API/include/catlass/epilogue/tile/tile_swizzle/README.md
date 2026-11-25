# tile_swizzle

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_swizzle.hpp)

[TOC]

## 概述

`tile_swizzle` 模块实现 epilogue 阶段的 Tile 遍历策略，控制 epilogue 计算中 tiles 的访问顺序。不执行计算，仅提供 tile 坐标和 shape 查询接口。

## API 清单

| API                                                                    | 遍历方式 | GetTileCoord(i)        | 说明         |
| :--------------------------------------------------------------------- | :------- | :--------------------- | :----------- |
| [EpilogueIdentityTileSwizzle](./epilogue_identity_tile_swizzle.md)     | 行列优先 | `(i / cols, i % cols)` | 默认策略     |
| [EpilogueHorizontalTileSwizzle](./epilogue_horizontal_tile_swizzle.md) | 水平优先 | `(i % rows, i / rows)` | 水平方向优先 |

## 遍历顺序对比

```cpp
blockShape(64, 128), tileShape(32, 64)

Identity:   loop0→(0,0)  loop1→(0,1)  loop2→(1,0)  loop3→(1,1)
Horizontal: loop0→(0,0)  loop1→(1,0)  loop2→(0,1)  loop3→(1,1)
```

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_swizzle.hpp"

using namespace Catlass::Epilogue::Tile;

MatrixCoord blockShape(64, 128);
MatrixCoord tileShape(32, 64);

EpilogueIdentityTileSwizzle swizzle(blockShape, tileShape);

uint32_t loops = swizzle.GetLoops();
for (uint32_t i = 0; i < loops; ++i) {
    MatrixCoord tileCoord = swizzle.GetTileCoord(i);
    MatrixCoord actualShape = swizzle.GetActualTileShape(tileCoord);
}
```
