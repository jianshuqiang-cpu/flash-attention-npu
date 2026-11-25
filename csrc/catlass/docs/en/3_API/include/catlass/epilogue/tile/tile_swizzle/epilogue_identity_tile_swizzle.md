# EpilogueIdentityTileSwizzle

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_swizzle.hpp)

[TOC]

## Function

`EpilogueIdentityTileSwizzle` implements a row-column-first tile traversal policy in the epilogue stage. Tiles are traversed in `(row, col)` order, with `GetTileCoord(i)` returning `(i / cols, i % cols)`.

- It applies to all architectures (no specialization).
- It performs no computation. It provides only tile coordinate and shape query APIs.

## Template Prototype

```cpp
struct EpilogueIdentityTileSwizzle;
```

There are no template parameters. `blockShape` and `tileShape` are passed in the constructor.

## Common APIs

| Method| Return Value| Description|
| :------ | :------ | :------ |
| `GetLoops()`| `uint32_t`| Returns the total number of tiles, calculated as `loopsMN.row() * loopsMN.column()`.|
| `GetTileCoord(loopIdx)`| `MatrixCoord`| Returns `(i / cols, i % cols)`.|
| `GetActualTileShape(tileCoord)`| `MatrixCoord`| Returns the actual tile shape. The boundary tile may be less than `tileShape`.|

## Traversal Order

```cpp
// blockShape(64, 128), tileShape(32, 64)
// loop 0 → (0,0), loop 1 → (0,1), loop 2 → (1,0), loop 3 → (1,1)
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
