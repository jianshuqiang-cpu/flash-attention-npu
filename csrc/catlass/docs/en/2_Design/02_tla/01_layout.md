# TLA Layouts

This document introduces `Layout` in Tensor Layout Abstraction (TLA).

If a Tensor is viewed as a logical multi-dimensional array, then `Layout` is responsible for answering the following questions:

- Which linear address corresponds to a logical coordinate `(i, j, ...)`
- How large is this Tensor logically
- Which positions contain logically valid data when the underlying memory has blocking, alignment, or padding

Therefore, `Layout` can be understood as a mapping rule from logical coordinates to memory addresses. Algorithms typically rely on this set of rules to access data, rather than directly depending on the underlying physical layout. This allows the same computation logic to adapt to ordinary ND layouts, row-major, column-major, and fractal layouts such as `zN` and `nZ`.

## Three Basic Concepts

### Logical Coordinate (coord)

A `coord` represents the position of an element in the logical space of a Tensor. The conventions are as follows:

- Coordinates start from 0.
- The unit of a coordinate is an element, not a byte or a tile index.
- The rank of a `coord` must match the logical dimension of the Tensor or Layout.
- Even when the underlying layout is `zN` or `nZ`, the `coord` still describes the logical row and column position, for example `(row, col)`.

For example, for a matrix with a logical shape of `(8, 16)`, `coord = (2, 4)` indicates the element in at row 2, column 4. It does not care whether this data is stored in row-major order, column-major order, or as fractal blocks in memory.

### Logical Shape and Memory Layout

In TLA, these two concepts are deliberately separated:

- Logical shape: The number of rows and columns of a Tensor from the user perspective.
- Memory layout: How these logical elements are arranged in memory and how many positions to skip when moving across a dimension.

The core value of `Layout` is to clearly express both the logical shape and the memory layout.

### Tail Tile

When a matrix size is not an integer multiple of the tile size, boundary tiles often contain only a subset of valid elements. Such boundary tiles are commonly called tail tiles.

TLA uses `originShape` to express the logically valid range. Therefore, users typically do not need to manually derive the actual size of each tail tile.

## Basic Types

### Tuple

TLA is built on [`tla::tuple`](../../../../include/tla/tuple.hpp). Its purpose is similar to `std::tuple`. Both represent a fixed-length sequence of elements. The difference is that TLA provides customizations for template metaprogramming and high-performance scenarios.

### IntTuple

[`IntTuple`](../../../../include/tla/int_tuple.hpp) is one of the most commonly used basic concepts in TLA. It can be:

- An integer, such as `int{2}` or `size_t{16}`
- A compile-time integer, such as `Int<3>{}` or its alias `_3`
- A tuple recursively composed of the preceding elements, such as `make_tuple(int{2}, Int<3>{})`

Therefore, `IntTuple` can represent either a one-dimensional size or a nested size with hierarchical structure.

Common operations include:

- `rank(IntTuple)`: Returns the number of elements.
- `get<I>(IntTuple)`: Returns the `I`-th element.
- `depth(IntTuple)`: Returns the number of nesting layers. The `depth` of a common integer is 0.

`IntTuple` is used not only for `Layout`, but also for types such as `Shape` and `Stride`. For details, see [`include/tla/layout.hpp`](../../../../include/tla/layout.hpp).

## Layout Composition

A `Layout` consists of three `IntTuple` objects: `Shape`, `Stride`, and `OriginShape`.

| Field         | Purpose                                             | Focus                                                                                     |
| ------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `Shape`       | Size description used for memory layout calculation | Determines the layout structure, which may not necessarily equal the actual logical size. |
| `Stride`      | Stride in each dimension                            | Determines how a coordinate maps to a linear address.                                     |
| `OriginShape` | Actual logical size of the Tensor                   | Determines which elements are logically valid.                                            |

These can be understood as:

- `Shape` describes how the memory is structured.
- `Stride` describes how far each step goes.
- `OriginShape` describes how much valid data there is logically.

`Shape` and `OriginShape` could be confusing. They do not duplicate each other:

- `Shape` is oriented towards layout calculation and allows for structures after alignment, tiling, or padding.
- `OriginShape`, designed for logical semantics, describes only the true valid data range.

![Origin_Shape-layout_1.png](https://raw.gitcode.com/user-images/assets/7631999/936388d2-81b6-400d-afe2-788eaf140f81/Origin_Shape-layout_1.png "Origin_Shape-layout_1.png")

`OriginShape` separates "how memory is structured" from "which data is logically valid."

- `Shape`: Designed for layout calculation. May include sizes after alignment, tiling, or padding.
- `OriginShape`: Designed for logical semantics. Describes the true valid data range.

For example, when a matrix with a logical size of `100 x 100` uses the `zN` layout, the following may occur:

- `originShape = (100, 100)`
- `shape = ((16, 7), (16, 7))`

The reason is:

- `16 * 7 = 112`, indicating that the underlying memory is organized as a blocked structure of `112 x 112`.
- However, logically, only `100 x 100` elements are valid.

This is also the foundation for TLA's ability to automatically handle tail tiles. Users at the block and kernel layers typically only need to program in terms of tiles. The valid boundary range is passed and cropped via `originShape`, eliminating the need to manually determine tail block conditions at each layer.

## Common Layout Interfaces

`Layout` provides a set of access interfaces consistent with the `IntTuple` style:

- `rank(Layout)`: Logical dimension of the layout.
- `get<I>(Layout)`: Obtains the `I`-th component.
- `depth(Layout)`: Number of nesting layers of the layout.
- `shape(Layout)`: Returns `Shape`.
- `stride(Layout)`: Returns `Stride`.
- `originShape(Layout)`: Returns `OriginShape`.

Recursive helper interfaces are also provided, for example:

- `get<I0, I1, ..., IN>(x)`: Recursively fetches a sub-unit.
- `rank<I...>(x)`: Returns the rank of a sub-unit.
- `depth<I...>(x)`: Returns the depth of a sub-unit.
- `shape<I...>(x)`: Returns the shape of a sub-unit.
- `originShape<I...>(x)`: Returns the origin shape of a sub-unit.

## Layout Construction

`Layout` supports construction using static integers, dynamic integers, and mixtures of both. It also supports ordinary matrix layouts and common internal layouts used in Ascend.

Common internal layouts in the Ascend cube core include `zN`, `nZ`, `zZ`, `nN`, and `L0C`. In scenarios such as GEMV, Scale, and Bias, one-dimensional `VectorLayout` is also used.

```c++
using namespace tla;

// 1. Directly provide the shape and stride. originShape is deduced by the system.
Layout w2xh4 = MakeLayout(MakeShape(Int<2>{}, 4),
                          MakeStride(Int<12>{}, Int<1>{}));

// 2. Nested layout. originShape is implicitly deduced to (16*2, 16*3) = (32, 48).
Layout w32xh48 = MakeLayout(MakeShape(MakeShape(16, 2), MakeShape(16, 3)),
                            MakeStride(MakeStride(16, 256), MakeStride(1, 512)));

// 3. Explicitly specify originShape.
Layout w2xh4_explicit = MakeLayout(MakeShape(Int<2>{}, 4),
                                   MakeStride(Int<12>{}, Int<1>{}),
                                   MakeShape(2, 4));

Layout w32xh48_explicit = MakeLayout(MakeShape(MakeShape(16, 2), MakeShape(16, 3)),
                                     MakeStride(MakeStride(16, 256), MakeStride(1, 512)),
                                     MakeShape(32, 48));

// 4. When rank=2, you can also use LayoutTag + (rows, cols) for construction.
auto rm = MakeLayout<float, Catlass::layout::RowMajor>(2, 4);

// 5. One-dimensional VectorLayout
auto vec = MakeLayout(128);
```

In the preceding information:

- `MakeLayout` returns a `Layout`.
- `MakeShape` returns a `Shape`.
- `MakeStride` returns a `Stride`.

The preceding layout can be written as follows:

```text
w2xh4   : (_2, 4):(_12, _1)
w32xh48 : ((16, 2), (16, 3)):((16, 256), (1, 512))
```

The interpretation is:

- The first part is `Shape`.
- The second part is `Stride`.
- If `OriginShape` is omitted, it can be deduced from `Shape` or is consistent with the logical size.

## Understanding Shape and Stride with Intuitive Examples

### 2x3 Row-Major

```text
shape  = (2, 3)
stride = (3, 1)
```

Meaning:

- Moving one step forward in the row dimension increases the linear address by 3.
- Moving one step forward in the column dimension increases the linear address by 1.

Therefore, the linear address order is:

| Logical Coordinate | Linear Address |
| ------------------ | -------------- |
| `(0, 0)`           | `0`            |
| `(0, 1)`           | `1`            |
| `(0, 2)`           | `2`            |
| `(1, 0)`           | `3`            |
| `(1, 1)`           | `4`            |
| `(1, 2)`           | `5`            |

### 2x3 Column-Major

```text
shape  = (2, 3)
stride = (1, 2)
```

Meaning:

- Moving one step forward in the row dimension increases the linear address by 1.
- Moving one step forward in the column dimension increases the linear address by 2.

Therefore, the linear address order is:

| Logical Coordinate | Linear Address |
| ------------------ | -------------- |
| `(0, 0)`           | `0`            |
| `(1, 0)`           | `1`            |
| `(0, 1)`           | `2`            |
| `(1, 1)`           | `3`            |
| `(0, 2)`           | `4`            |
| `(1, 2)`           | `5`            |

### Understanding Nested Layouts Using `zN` as an Example

Example layout:

```text
shape  = ((4, 2), (4, 3))
stride = ((4, 16), (1, 32))
```

That is,

- In the row direction, first take an inner block of size 4, then repeat 2 times along the row direction.
- In the column direction, first take an inner block of size 4, then repeat 3 times along the column direction.
- How to move within a sub-block and how to jump between sub-blocks are defined by the nested `Stride`.

The key point is not to memorize every number, but to understand that TLA uses nested `Shape` and `Stride` to explicitly express the structural hierarchy of a tiled layout, rather than hardcoding such formats into algorithms.

## How Coordinates Map to Indexes

In TLA, you can use `tla::crd2offset(coord, shape, stride)` to convert logical coordinates into linear indexes.

Constraints:

- The ranks of `coord`, `shape` and `stride` must be the same.
- `coord` represents the logical element coordinate, not a byte offset.

```cpp
auto shape  = Shape<Shape<_4, _2>, Shape<_4, _3>>{};
auto stride = Stride<Stride<_4, _16>, Stride<_1, _32>>{};

print(crd2offset(tla::MakeCoord(1, 5), shape, stride));  // 37
```

This code indicates that in a matrix with a logical size of `(8, 12)` and a underlying fractal layout, the logical coordinate `(1, 5)` maps to linear index `37`.

## Obtaining TileLayout

TileLayout can be obtained through `GetTileLayout`:

```cpp
template <class Layout, class TileShape, class Coord>
auto GetTileLayout(Layout const& layout,
                   TileShape const& tileShape,
                   Coord const& coord);

using namespace tla;
Layout a = Layout<Shape<Shape<_4, _2>, Shape<_4, _3>>,
                  Stride<Stride<_4, _16>, Stride<_1, _32>>,
                  Shape<_8, _12>>{};

Layout a0 = GetTileLayout(a, MakeShape(4, 4), MakeCoord(6, 10));
// The stride remains unchanged, and the logical valid range is cropped to (2, 2).
```

Parameter semantics:

- `tileShape`: size of the tile to extract, in elements.
- `coord`: element coordinate of the tile's upper left corner in the parent layout's logical space, in elements.

That is, `coord = (6, 10)` means extracting tiles starting from logical row 6 and column 10, not the sixth tile and the tenth tile.

### Core Semantics of `GetTileLayout`

`GetTileLayout` returns the `Layout` of a tile view and does not change the underlying data layout. It completes three things:

1. Preserves the original `stride()` because the underlying memory layout has not changed.
2. Constructs the tile's `shape()` using `tileShape`. When the parent layout has a nested structure, the returned result maintains the same structural hierarchy if necessary.
3. Calculates the tile's `originShape()` based on the parent layout's `originShape()` and start `coord`.

Among these, step 3 is the most critical:

$$
origin\_shape[d] = \min(tileShape[d], \max(origin\_base[d] - coord[d], 0))
$$

It indicates how many valid elements logically remain starting from the current position. Therefore:

- For tiles in the interior region, `originShape == tileShape`.
- For tail tiles, `originShape` automatically shrinks.

### What Does "Convert to the corresponding `shape()` based on the parent layout's structure" Mean?

This means that when the parent layout itself is a nested layout, the tile's `shape()` must also maintain the same structural hierarchy so that subsequent access rules can continue to be reused.

For example, the rows and columns of the parent layout are organized with inner blocks of size `16`:

```text
parent shape = ((16, 7), (16, 7))
parent originShape = (100, 100)
```

If you want to extract a tile with a logical size of `(32, 48)`, you can write the size as `(32, 48)`. However, given the parent layout is `zN`, the corresponding `shape()` will be expressed in the structure of the parent layout as follows:

```text
tile logical size = (32, 48)
tile shape         = ((16, 2), (16, 3))
```

What happens here is a structure conversion, not a data rearrangement:

- Logically, the tile is still `32 x 48`.
- In terms of layout, it is expressed as an inner block of size 16 in each dimension, multiplied by the number of outer blocks.
- `stride()` is still inherited from the parent layout, so access rules remain unchanged.

This ensures that the parent layout and tile layout maintain a consistent structural hierarchy.

![Origin_Shape-layout_2.png](https://raw.gitcode.com/user-images/assets/7631999/649c84f3-981f-49eb-be77-6cbf6fd1e5b3/Origin_Shape-layout_2.png "Origin_Shape-layout_2.png")

### Parameter Constraints

- Both `tileShape` and `coord` must be tuples at the top level, that is, `depth == 1`.
- `rank(coord) == rank(tileShape)`

### Behavior in Different Layouts

- If the parent layout is a common vector or matrix, the returned `shape()` of the layout is typically equal to `tileShape`.
- If the parent layout is a nested or fractal layout, such as `zN`, `nZ`, `zZ`, or `L0C`, the current implementation supports only `rank == 2` and converts the `tileShape` in `(rows, cols)` format into a nested `Shape` with the same structure as the parent layout.
