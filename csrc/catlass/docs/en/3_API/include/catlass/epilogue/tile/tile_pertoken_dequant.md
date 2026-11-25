# TilePerTokenDequant

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_pertoken_dequant.hpp)

[TOC]

## Function

`TilePerTokenDequant` implements per-token dequantization in the epilogue stage. It multiplies the int32 accumulation results in Unified Buffer by both the per-channel scale and the per-token scale, dequantizing them into the target floating-point type.

- Applicability: `Arch::Ascend950` only
- Style: TLA (operands are encapsulated using `tla::Tensor`, and micro-architecture intrinsic instructions are used internally.)
- Computing process: `dst[i,j] = (int32)src[i,j] * (float)scale[j] * (float)perToken[i]`
- `__simd_vf__` inline assembly optimization for int32 →float32 type conversion directly in the register

## Template Prototype

```cpp
template <
    class ArchTag_,          // Architecture tag (static assertion for Ascend 950 only)
    class ElementSrc_,       // Source element type (static assertion for int32_t)
    class ElementScale_,     // Per-channel scale element type (half/bfloat16_t/float)
    class ElementPerToken_,  // Per-token scale element type (half/bfloat16_t/float)
    class ElementDst_,       // Destination element type (half/bfloat16_t/float)
    class TileShape_         // Tile shape type (including COLUMN)
>
struct TilePerTokenDequant;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Only `Arch::Ascend950` is supported, with compile-time assertion checks.|
| `ElementSrc_` | Source element type. Only `int32_t` is supported.|
| `ElementScale_` | Per-channel scale element type, including `half`/`bfloat16_t`/`float`|
| `ElementPerToken_` | Per-token scale element type, including `half`/`bfloat16_t`/`float`|
| `ElementDst_` | Destination element type, including `half`/`bfloat16_t`/`float`|
| `TileShape_` | Tile shape. `TileShape_::COLUMN` is used to determine N_BASE_SIZE.|

## APIs

```cpp
template <class TensorDst, class TensorSrc, class TensorScale, class TensorPerToken>
void operator()(
    TensorDst const &ubOut,               // Destination Unified Buffer TLA tensor (RowMajor)
    TensorSrc const &ubIn,                // Source Unified Buffer TLA tensor (RowMajor, int32_t)
    TensorScale const &ubScale,           // Per-channel scale TLA tensor (VectorLayout)
    TensorPerToken const &ubPerToken      // Per-token scale TLA tensor (VectorLayout)
)
```

| Parameter| Description|
| :------ | :------ |
| `ubOut` | Unified Buffer TLA tensor, type `ElementDst`, layout `RowMajor`|
| `ubIn` | Unified Buffer TLA tensor, type `int32_t`, layout `RowMajor`, MMAD accumulation output|
| `ubScale` | Per-channel scale, type `ElementScale`, layout `VectorLayout` (status=0), length = n|
| `ubPerToken` | Per-token scale, type `ElementPerToken`, layout `VectorLayout` (status=0), length = m|

Static assertions ensure that all tensors have their position set to `VECCALC`, and that their layouts match.

## Examples

```cpp
#include "catlass/epilogue/tile/tile_pertoken_dequant.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128;
constexpr uint32_t N = 256;

using TileShape = Shape<M, N>;

using DequantOp = TilePerTokenDequant<Arch::Ascend950, int32_t, half, half, half, TileShape>;

auto srcLayout = tla::MakeLayout<int32_t, layout::RowMajor>(M, N);
auto scaleLayout = tla::MakeLayout<half, layout::VectorLayout>(N, 1);
auto perTokenLayout = tla::MakeLayout<half, layout::VectorLayout>(M, 1);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<int32_t> ubIn;
AscendC::LocalTensor<half> ubScale, ubPerToken, ubOut;

auto srcTensor = tla::MakeTensor(ubIn, srcLayout, Arch::PositionUB{});
auto scaleTensor = tla::MakeTensor(ubScale, scaleLayout, Arch::PositionUB{});
auto perTokenTensor = tla::MakeTensor(ubPerToken, perTokenLayout, Arch::PositionUB{});
auto dstTensor = tla::MakeTensor(ubOut, dstLayout, Arch::PositionUB{});

DequantOp dequantOp;
dequantOp(dstTensor, srcTensor, scaleTensor, perTokenTensor);
```
