# TLA Tensors

This document introduces `Tensor` in Tensor Layout Abstraction (TLA).

If `Layout` is responsible for describing how logical coordinates map to memory, then `Tensor` is an accessible object that binds concrete data, the current view starting point, and the storage hierarchy on top of a `Layout`.

In this document, `Tensor` always refers to a logical view:

- `MakeTensor` creates a view, and no data copy occurs.
- The slicing result of `operator()` is a subview, and no data copy occurs.
- `GetTile` and `TileView` return tile views, and no data copy occurs.
- `MakeTensorLike` only binds an existing storage to a new view whose logical size is the same as that of the reference tensor. It does not perform data movement.

Actual data movement should be performed by explicit movement or computation interfaces, not implicitly by these view construction interfaces.

For the basic definition of `Layout`, see [Layout](./01_layout.md).

## Distinguishing Four Components

The template parameters of `Tensor` are `BuiltinTensor`, `Layout`, `Coord`, and `Position`. It is recommended that you learn them separately.

### BuiltinTensor

`BuiltinTensor` is an underlying tensor object provided by Ascend C, such as `GlobalTensor` or `LocalTensor`. It represents the underlying storage object itself.

### Layout

`Layout` describes how logical coordinates map to memory and how the logical valid range is expressed.

### Coord

`Coord` is the start coordinate of the current `Tensor` view in the parent logical space expressed by BuiltinTensor.

Two points need special emphasis here:

- The unit of `coord` is element, not byte.
- `coord` indicates where this view starts from the parent logical space expressed by BuiltinTensor, not the tile index.

For example, in a matrix with a logical size of `(8, 16)`, if the `coord()` of a sub-tensor is `(2, 4)`, it indicates that the upper left corner of this view corresponds to row 2 and column 4 of the parent logical matrix.

### Position

`Position` is a location tag in Ascend C, such as `Arch::PositionGM{}` and `Arch::PositionL1{}`. It distinguishes which level of storage (GM, L1, L0, etc.) the data resides in.

## Tensor Construction

Currently, `MakeTensor` is used to construct `Tensor`.

```cpp
using namespace tla;
GlobalTensor<float> A = ...;

auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);

// 1. Start from logical coordinate (0, 0) by default.
auto tensorA = MakeTensor(A, layout, Arch::PositionGM{});

// 2. Explicitly specify the current view's starting point.
auto tensorA_sub = MakeTensor(A, layout, tla::MakeCoord(1, 5), Arch::PositionGM{});
```

You can understand it in the following way:

- `layout` determines how to interpret the memory.
- `coord` determines where the current view starts in the parent logical space expressed by the BuiltinTensor.

## Common Tensor Interfaces

TLA `Tensor` provides the following common interfaces:

- `.data()`: returns the underlying memory object.
- `.layout()`: returns the layout.
- `.coord()`: returns the starting point of the current view.
- `.shape()`: returns `layout.shape()`.
- `.stride()`: returns `layout.stride()`.
- `.originShape()`: returns `layout.originShape()`.
- `(coord0, coord1, ...)`: indexes or slices by coordinates.

## Understanding of Three Types of Coordinates

The most confusing part in the TLA document is the different types of coordinates. The following provides a unified convention.

### Element Coordinates (element coord)

Element coordinates represent the logical position of an element, for example, `(row, col)`. The coordinates used by interfaces such as `GetTile`, `crd2offset`, and common index access are of this type.

### Tile Coordinates (tile coord)

Tile coordinates indicate the tile index, not the element index. For example, in `tileShape = (64, 128)`:

- `tileCoord = (1, 2)` indicates the first row tile and the second column tile.
- Its corresponding element starting point is `(1 * 64, 2 * 128)`.

### View Coordinates (view coord)

`tensor.coord()` indicates the starting point of the current `Tensor` view in the parent logical space expressed by the BuiltinTensor. It is determined by the operation that creates the view, such as `MakeTensor`, `GetTile`, `TileView`, or slicing.

It can be summarized as follows:

- `element coord` is the element position.
- `tile coord` is the tile index.
- `tensor.coord()` is the starting point of the current view.

## Understanding `coord()` with a Complete Example

The following uses the same matrix to describe the `MakeTensor` and `GetTile` scenarios.

```cpp
using namespace tla;

GlobalTensor<float> A = ...;
auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);

auto tensorA = MakeTensor(A, layout, Arch::PositionGM{});
// tensorA.coord() == (0, 0)

auto tensorA_sub = MakeTensor(A, layout, MakeCoord(1, 5), Arch::PositionGM{});
// tensorA_sub.coord() == (1, 5)

auto tileA = GetTile(tensorA_sub, MakeCoord(2, 4), MakeShape(4, 8));
// tileA.coord() == (3, 9)
```

From the above:

1. `tensorA` directly observes the entire logical matrix, so the starting point is `(0, 0)`.
2. `tensorA_sub` observes `(1, 5)` of the parent logical space expressed by the BuiltinTensor, so the starting point changes to `(1, 5)`.
3. `tileA` takes a tile whose starting point is `(2, 4)` based on `tensorA_sub`. Therefore, the starting point of the new view is `(1, 5) + (2, 4) = (3, 9)`.

![Origin_Shape-tensor_1.png](https://raw.gitcode.com/user-images/assets/7631999/004cd08b-cda4-4c99-a5c1-63770d71f151/Origin_Shape-tensor_1.png "Origin_Shape-tensor_1.png")

## Using `operator()` for Indexing and Slicing

TLA `Tensor` supports indexing using `operator()` and supports using `tla::_` to express full-dimensional slicing, returning a sub-tensor view.

### Basic Rules

- If there is no `tla::_`, `tensor(i, j, ...)` returns an underlying `BuiltinTensor` access result, which is essentially equivalent to `tensor.data()[offset]`.
- With `tla::_`, `tensor(..., tla::_, ...)` returns the sub-tensor view. The indexed dimension is fixed, and the dimension where `tla::_` is located is retained.
- The coordinate parameters used here must be one-level tuples, that is, each dimension is a scalar or `tla::_`. Nested tuples are not supported.

The equivalent semantics can be written as follows:

```cpp
tensor.data()[tensor.layout()(tensor.coord() + coord_arg)]
```

### Output Tensor Dimensions

Assume that the rank of the input tensor is $R$, and the set of dimension indices where `tla::_` appears in `coord` is $\{d_0, d_1,..., d_{k-1}\}$. Then:

- The rank of the output tensor is $k$.
- The `layout.shape()`, `layout.stride()` and `layout.originShape()` of the output tensor are the projections of the input layout in these dimensions.
- The `coord()` of the output tensor will start from all zeros again because it has become a new local view.

For example, for a 3D tensor `A(B, M, K)`:

```cpp
auto A2 = A3(b, tla::_, tla::_);  // 3D -> 2D. The (M, K) view is obtained.
auto A1 = A2(r, tla::_) // 2D -> 1D. The (K) view is obtained.
```

![Origin_Shape-tensor_2.png](https://raw.gitcode.com/user-images/assets/7631999/dd1a947a-2371-4e6e-9fef-51a5ceb1556a/Origin_Shape-tensor_2.png "Origin_Shape-tensor_2.png")

## Obtaining TileTensor

### GetTile

`GetTile` is used to obtain a tile view from the parent tensor without copying data.

```cpp
template <class Tensor, class Coord, class Shape>
auto GetTile(Tensor const& tensor,
             Coord const& coord,
             Shape const& shape);
```

Parameter semantics:

- `coord`: element coordinates, indicating the starting point of the tile's upper left corner in the logical space of the parent tensor.
- `shape`: expected tile size, in the unit of element.

```cpp
using namespace tla;

auto layout = tla::MakeLayout<float, Catlass::layout::RowMajor>(8, 16);
auto tensor = MakeTensor(A, layout, Arch::PositionGM{});

// Take a 4 x 8 tile starting from the logical coordinates (2, 4).
auto tile = GetTile(tensor, tla::MakeCoord(2, 4), MakeShape(4, 8));
```

The returned result can be interpreted as follows:

- `tile.coord()` = `tensor.coord()` + `(2, 4)`
- `tile.layout().shape()` indicates the expected tile size or the expression consistent with the parent layout structure.
- `tile.layout().originShape()` indicates the actual logical range of the tile, automatically cropped at edges.

### Constraints

- `tensor.layout().depth == 1` is supported.
- In the case of `tensor.layout().depth > 1`, that is, a fractal or nested layout, the current `GetTileLayout` supports only `rank == 2`.
- Both `coord` and `shape` must be one-layer tuples and satisfy `rank(coord) == rank(shape) == Tensor::rank`.

### Edge Behavior

For example, if the logical size of the parent tensor is `(8, 16)`, run the following command:

```cpp
auto tail = GetTile(tensor, tla::MakeCoord(6, 10), MakeShape(4, 8));
```

Then:

- The expected size is still `(4, 8)`.
- However, logically, only two rows and six columns of valid data remain.
- Therefore, `tail.layout().originShape()` becomes `(2, 6)`.

### TileView

The behavior of `TileView` is equivalent to that of `GetTile`. The only difference is that the units of the input coordinates are different.

- `GetTile` accepts element coordinates.
- `TileView` accepts tile coordinates.

```cpp
template <class TensorT, class TileCoord, class TileShape>
auto TileView(TensorT const& tensor,
              TileCoord const& tileCoord,
              TileShape const& tileShape);
```

Example:

```cpp
auto tensorTileA = tla::TileView(
      tensorA,
      tla::MakeCoord(0u, kLoopIdx),
      tla::MakeShape(Int<L1_TILE_M>{}, Int<L1_TILE_K>{})
);
```

### Equivalence

The `TileView` and `GetTile` can be directly understood according to the following equations:

```cpp
TileView(t, tileCoord, tileShape) = GetTile(t, tileCoord ⊙ tileShape, tileShape)
```

Here, `⊙` indicates element-wise multiplication. For example:

```cpp
(1, 2) ⊙ (64, 128) = (64, 256)
```

This equation indicates that:

1. `TileView` converts the tile coordinates into element coordinates first.
2. Then, the same tile view is created according to the rules of `GetTile`.

Therefore, the difference between the two lies only in the coordinate unit provided by the caller, rather than the logical semantics of the returned result.

![Origin_Shape-tensor_3.png](https://raw.gitcode.com/user-images/assets/7631999/b543ee2b-c938-4e19-87b8-40abd0c81b53/Origin_Shape-tensor_3.png "Origin_Shape-tensor_3.png")

### Why `TileView` Is More Suitable for Tile Looping

In actual kernel or block loops, loop variables are typically tile indices rather than element coordinates. Therefore, `TileView` is more direct.

The following uses the same example of K-dimensional tiling for comparison.

#### Method 1: Using `GetTile`

```cpp
constexpr uint32_t tileM = 64;
constexpr uint32_t tileK = 128;

for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
      auto coord = tla::MakeCoord(0u, kTile * tileK);
      auto shape = tla::MakeShape(tileM, tileK);
      auto tensorTileA = tla::GetTile(tensorA, coord, shape);
      // use tensorTileA
}
```

#### Method 2: Using `TileView`

```cpp
constexpr uint32_t tileM = 64;
constexpr uint32_t tileK = 128;

for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
      auto tensorTileA = tla::TileView(
            tensorA,
            tla::MakeCoord(0u, kTile),
            tla::MakeShape(tileM, tileK)
      );
      // use tensorTileA
}
```

The logical results of these two code snippets are the same, but the second method directly uses tile coordinates, which is closer to the semantics of the tile loop itself and makes it less likely to confuse tile coordinates with element coordinates.

## Creating a Similar Tensor

### MakeTensorLike

`MakeTensorLike` is used to create a new tensor whose logical size is the same as that of the `likeTensor`. The most common use case is: starting from an existing tile view, construct a corresponding Tensor in another level of memory, automatically inheriting its `originShape()`.

When layoutBase is not specified, the behavior is to decide the layout based on LayoutTagDst, infer ElementDst from LikeTensor::Element, extract the size from likeTensor's originShape, and call MakeLayout<ElementDst, LayoutTagDst>(originShape()) to construct the target layout (which may round up the shape at fractal granularity due to fractal layout legality requirements).

If layoutBase is specified, use MakeLayout(layoutBase.shape(), layoutBase.stride(), likeTensor.originShape()) to construct the target layout.

Again, it must be emphasized: `MakeTensorLike` constructs a new view and does not perform data movement. It simply binds the user-provided `builtinTensor` as a new TLA `Tensor` and makes this new view reuse the logical size semantics of `likeTensor`.

Currently, `MakeTensorLike` supports only `likeTensor.rank <= 2`.

The interface serves three typical scenarios.

```cpp
// (1) Infer ElementDst from LikeTensor::Element.
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType);

// (2) Explicitly specify ElementDst.
template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType);

// (3) Provide layoutBase.
template <class LayoutTagDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType,
                    LayoutBase const& layoutBase);

template <class LayoutTagDst, class ElementDst, class BuiltinTensor, class LikeTensor, class PositionType, class LayoutBase>
auto MakeTensorLike(BuiltinTensor const& builtinTensor,
                    LikeTensor const& likeTensor,
                    PositionType,
                    LayoutBase const& layoutBase);
```

### Scenario 1: Source and Destination Element Types Are the Same

This is the most common scenario. For example, a `half` tile in GM is used to create a corresponding L1 tensor. The element type remains unchanged, and only the storage hierarchy changes.

```cpp
auto tensorTileA = tla::TileView(
      tensorA,
      tla::MakeCoord(blockM, kTile),
      tla::MakeShape(L1_TILE_M, L1_TILE_K)
);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensorList[l1ListId],
      tensorTileA,
      Arch::PositionL1{}
);

// Result:
// 1. tensorL1A uses the L1 target layout.
// 2. The originShape of tensorL1A is the same as that of tensorTileA.
// 3. The element type is automatically inferred from likeTensor.
```

### Scenario 2: Target Element Type Is Different

If the element type of the target tensor is different from that of the source tensor, you need to explicitly specify `ElementDst`. Example:

- Use an accumulator type in L0C.
- Generate a `float` accumulation view from the `half` input.
- The `PrimType` of the target memory object is different from that of the `LikeTensor::Element`.

```cpp
auto tensorL0C = tla::MakeTensorLike<LayoutTagL0C, float>(
      l0cTensor,
      tensorTileC,
      Arch::PositionL0C{}
);

// Result:
// 1. The logical size of tensorL0C is inherited from tensorTileC.
// 2. The target element type is explicitly float.
// 3. This applies to the accumulator or type promotion scenarios.
```

### Scenario 3: Additional Control Over the Target Layout Is Needed

In certain scenarios, only specifying `LayoutTagDst` is not enough because the base shape or stride of the target layout needs to be explicitly provided by the user. Example:

- The target tensor uses a specific fractal layout.
- The physical layout of a certain L1 needs to be fixed. Note that the layout of L0 is uniquely determined by originShape. Therefore, customizing an unexpected layout on L0 is invalid.
- A special `shape/stride` structure needs to be provided in advance, but the logical valid range still needs to be inherited from `likeTensor`.

```cpp
auto layoutBaseL1A = tla::MakeLayout<half, LayoutTagL1A>(L1_TILE_M, L1_TILE_K);

auto tensorL1A = tla::MakeTensorLike<LayoutTagL1A>(
      l1ATensor,
      tensorTileA,
      Arch::PositionL1A{},
      layoutBaseL1A
);

// Result:
// 1. The shape/stride of tensorL1A is from layoutBaseL1A.
// 2. The originShape of tensorL1A is inherited from tensorTileA in GM.
// 3. Even if the current tile is a tail tile, the logical valid range is not lost.
```

If both controlling the target layout and explicitly specifying the target element type are needed, the overload with both `layoutBase` and `ElementDst` can be used.

## Practical Usage Pattern

At the block and kernel layers, the common method is as follows:

1. Use `TileView` to obtain the tile view from the parent tensor, and edges are automatically processed.
2. Use `MakeTensorLike` to construct the corresponding tensor at the target memory hierarchy, which automatically inherits `originShape()`.

The benefits of this pattern are:

- The main process always revolves around tile programming.
- Tail tile logic is automatically passed through `originShape`.
- Both data movement and computation stages can reuse the same logical size semantics, reducing edge branches and ambiguity.
