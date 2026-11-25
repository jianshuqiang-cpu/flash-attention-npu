# EpilogueHorizontalTileSwizzle

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_swizzle.hpp)

[TOC]

## 功能说明

`EpilogueHorizontalTileSwizzle` 实现 epilogue 阶段的水平优先 tile 遍历策略。与 `EpilogueIdentityTileSwizzle` 接口相同，遍历顺序不同：按 column 顺序遍历，`GetTileCoord(i)` 返回 `(i % rows, i / rows)`。

- 适用范围：所有架构（无架构特化）
- 不执行计算，仅提供 tile 坐标和 shape 查询接口

## 模板原型

```cpp
struct EpilogueHorizontalTileSwizzle;
```

无模板参数，在构造函数中传入 `blockShape` 和 `tileShape`。

## 公共接口

| 方法                            | 返回值        | 说明                                                |
| :------------------------------ | :------------ | :-------------------------------------------------- |
| `GetLoops()`                    | `uint32_t`    | 返回总 tile 数 = `loopsMN.row() * loopsMN.column()` |
| `GetTileCoord(loopIdx)`         | `MatrixCoord` | 返回 `(i % rows, i / rows)`                         |
| `GetActualTileShape(tileCoord)` | `MatrixCoord` | 返回实际 tile shape（边界可能小于 `tileShape`）     |

## 遍历顺序

```cpp
// blockShape(64, 128), tileShape(32, 64)
// loop 0 → (0,0), loop 1 → (1,0), loop 2 → (0,1), loop 3 → (1,1)
```

## 调用示例

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
