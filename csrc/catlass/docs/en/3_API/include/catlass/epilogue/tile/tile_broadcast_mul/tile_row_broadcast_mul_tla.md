# TileRowBroadcastMulTla

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## Function

`TileRowBroadcastMulTla` implements the TLA-style broadcast multiplication operation in the epilogue stage. A row vector (1,  n) in Unified Buffer is broadcast to an (m, n) matrix and then multiplied element-wise with the input. `AscendC::Mul` is called after the offset is computed using `ubOut.layout()(ubOut.coord())`.

- Applicability: all architectures (no architecture specialization)
- Style: TLA

## Template Prototype

```cpp
template <
    class ArchTag_,        // Architecture tag
    class ElementCompute_, // Computation element type (directly passed in, not GemmType)
    class TileShape_       // Tile shape
>
struct TileRowBroadcastMulTla;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ElementCompute_` | Computation element type, for example, `half`|
| `TileShape_` | Tile shape, `Shape<ROW, COLUMN>`|

## APIs

```cpp
template <class TensorUbOut, class TensorUbIn0, class TensorUbIn1>
void operator()(TensorUbOut const &ubOut, TensorUbIn0 const &ubIn0, TensorUbIn1 const &ubIn1)
```

Call `AscendC::Mul` after computing the offset using `ubOut.layout()(ubOut.coord())`.

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128, N = 256;

auto layout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<half> ubOutData, ubIn0Data, ubIn1Data;
auto ubOut = tla::MakeTensor(ubOutData, layout, Arch::PositionUB{});
auto ubIn0 = tla::MakeTensor(ubIn0Data, layout, Arch::PositionUB{});
auto ubIn1 = tla::MakeTensor(ubIn1Data, layout, Arch::PositionUB{});

TileRowBroadcastMulTla<Arch::AtlasA2, half, Shape<M, N>> op;
op(ubOut, ubIn0, ubIn1);
```
