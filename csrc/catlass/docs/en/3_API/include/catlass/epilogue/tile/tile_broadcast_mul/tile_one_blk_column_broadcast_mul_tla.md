# TileOneBlkColumnBroadcastMulTla

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## Function

`TileOneBlkColumnBroadcastMulTla` implements the TLA-style one-block-column broadcast multiplication operation. Functionally identical to `TileOneBlkColumnBroadcastMul`, it is implemented through `tla::Tensor` encapsulation.

- Applicability: all architectures (no architecture specialization)
- Style: TLA

## Template Prototype

```cpp
template <
    class ArchTag_,        // Architecture tag
    class ElementCompute_, // Computation element type
    class TileShape_       // Tile shape
>
struct TileOneBlkColumnBroadcastMulTla;
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

TileOneBlkColumnBroadcastMulTla<Arch::AtlasA2, half, Shape<M, N>> op;
op(ubOut, ubIn0, ubIn1);
```
