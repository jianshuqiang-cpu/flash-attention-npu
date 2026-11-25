# EpilogueHorizontalTileSwizzle

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_swizzle.hpp)

[TOC]

## Function

`EpilogueHorizontalTileSwizzle` implements a horizontal-first tile traversal policy in the epilogue stage. Functionally similar to the `EpilogueIdentityTileSwizzle` API, it differs in traversal order: tiles are traversed by column, with `GetTileCoord(i)` returning `(i % rows, i / rows)`.

- It applies to all architectures (no specialization).
- It performs no computation. It provides only tile coordinate and shape query APIs.

## Template Prototype

```cpp
struct EpilogueHorizontalTileSwizzle;
```

There are no template parameters. `blockShape` and `tileShape` are passed in the constructor.

## Common APIs

| Method| Return Value| Description|
| :------ | :------ | :------ |
| `GetLoops()`| `uint32_t`| Returns the total number of tiles, calculated as `loopsMN.row() * loopsMN.column()`.|
| `GetTileCoord(loopIdx)` | `MatrixCoord` | Returns `(i % rows, i / rows)`.|
| `GetActualTileShape(tileCoord)`| `MatrixCoord`| Returns the actual tile shape. The boundary tile may be less than `tileShape`.|

## Traversal Order

```cpp
// blockShape(64, 128), tileShape(32, 64)
// loop 0 → (0,0), loop 1 → (1,0), loop 2 → (0,1), loop 3 → (1,1)
```

## Examples

```cpp
#include "catlass/epilogue/tile/tile_swizzle.hpp"

using namespace Catlass::Epilogue::Tile;

MatrixCoord blockShape(64, 128);
MatrixCoord tileShape(32, 64);

EpilogueHorizontalTileSwizzle swizzle(blockShape, tileShape);

uint32_t loops = swizzle.GetLoops();
for (uint32_t i = 0; i < loops; ++i) {
    MatrixCoord tileCoord = swizzle.GetTileCoord(i);
    MatrixCoord actualShape = swizzle.GetActualTileShape(tileCoord);
}
```
