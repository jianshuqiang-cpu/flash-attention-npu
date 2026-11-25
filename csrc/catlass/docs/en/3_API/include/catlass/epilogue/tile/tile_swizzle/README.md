# tile_swizzle

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_swizzle.hpp)

[TOC]

## Overview

The `tile_swizzle` module defines the tile traversal policy in the epilogue stage, controlling the access sequence of tiles during epilogue computation. It performs no computation itself and provides only tile coordinate and shape query APIs.

## API List

| API | Traversal Policy | GetTileCoord(i) | Description |
| :------ | :------ | :------ | :------ |
| [EpilogueIdentityTileSwizzle](./epilogue_identity_tile_swizzle.md) | Row-column first | `(i / cols, i % cols)` | Default policy |
| [EpilogueHorizontalTileSwizzle](./epilogue_horizontal_tile_swizzle.md) | Horizontal-first | `(i % rows, i / rows)` | Horizontal traversal first |

## Traversal Order Comparison

```bash
blockShape(64, 128), tileShape(32, 64)

Identity:   loop0→(0,0)  loop1→(0,1)  loop2→(1,0)  loop3→(1,1)
Horizontal: loop0→(0,0)  loop1→(1,0)  loop2→(0,1)  loop3→(1,1)
```

## Examples

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
