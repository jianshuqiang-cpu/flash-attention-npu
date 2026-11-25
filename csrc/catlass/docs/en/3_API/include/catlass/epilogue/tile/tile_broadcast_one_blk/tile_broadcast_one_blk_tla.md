# TileBroadcastOneBlkTla

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## Function

`TileBroadcastOneBlkTla` implements the TLA-style one-block broadcast operation. Functionally identical to `TileBroadcastOneBlk`, it is implemented through `tla::Tensor` encapsulation.

- Applicability: all architectures (no architecture specialization)
- Style: TLA

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ElementCompute_,    // Computation element type (directly passed in)
    uint32_t COMPUTE_LENGTH_  // Computation length
>
struct TileBroadcastOneBlkTla;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ElementCompute_` | Computation element type, for example, `half`|
| `COMPUTE_LENGTH_` | Total number of elements to be broadcast|

## APIs

```cpp
template <class TensorUbOut, class TensorUbIn>
void operator()(TensorUbOut &ubOut, TensorUbIn &ubIn)
```

Call `AscendC::Brcb` after computing the offset using `ubOut.layout()(ubOut.coord())`.

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"

using namespace Catlass::Epilogue::Tile;
constexpr uint32_t COMPUTE_LENGTH = 256;

auto layoutOut = tla::MakeLayout<half, layout::RowMajor>(COMPUTE_LENGTH, 32);
auto layoutIn = tla::MakeLayout<half, layout::VectorLayout>(COMPUTE_LENGTH, 1);

AscendC::LocalTensor<half> ubOutData, ubInData;
auto ubOut = tla::MakeTensor(ubOutData, layoutOut, Arch::PositionUB{});
auto ubIn = tla::MakeTensor(ubInData, layoutIn, Arch::PositionUB{});

TileBroadcastOneBlkTla<Arch::AtlasA2, half, COMPUTE_LENGTH> op;
op(ubOut, ubIn);
```
