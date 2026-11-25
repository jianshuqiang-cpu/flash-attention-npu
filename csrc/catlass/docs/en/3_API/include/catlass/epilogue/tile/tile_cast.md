# TileCast

> [Code Location](../../../../../../../include/catlass/epilogue/tile/tile_cast.hpp)

[TOC]

## Function

`TileCast` implements type conversion in the epilogue stage, converting the source data types in Unified Buffer to destination data types. It performs element-level type conversion using Ascend C's `Cast` instruction, applicable to scenarios such as floating-point quantization and dequantization.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA, directly operating on `AscendC::LocalTensor`
- Type conversion: determined by the `Element` members of `DstType_` and `SrcType_`

## Template Prototype

```cpp
template <
    class ArchTag_,       // Architecture tag
    class DstType_,       // Destination data type (including Element)
    class SrcType_,       // Source data type (including Element)
    class TileShape_      // Tile shape (including COUNT)
>
struct TileCast;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag, which is usually `Arch::AtlasA2` or `Arch::Ascend950`|
| `DstType_` | Destination data type. The element type is obtained through `DstType_::Element`.|
| `SrcType_` | Source data type. The element type is obtained through `SrcType_::Element`.|
| `TileShape_` | Tile shape type. The total number of elements is obtained through `TileShape_::COUNT`.|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> const &ubOut,  // Destination Unified Buffer LocalTensor
    AscendC::LocalTensor<ElementSrc> const &ubIn    // Source Unified Buffer LocalTensor
)
```

| Parameter| Description|
| :------ | :------ |
| `ubOut` | Destination Unified Buffer tensor of the `DstType_::Element` type|
| `ubIn` | Source Unified Buffer tensor of the `SrcType_::Element` type|

Type conversion is implemented internally through `AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT)`.

## Examples

```cpp
#include "catlass/epilogue/tile/tile_cast.hpp"

using namespace Catlass::Epilogue::Tile;

using SrcType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DstType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using CastOp = TileCast<Arch::AtlasA2, DstType, SrcType, TileShape>;

AscendC::LocalTensor<int32_t> srcTensor;
AscendC::LocalTensor<half> dstTensor;

CastOp castOp;
castOp(dstTensor, srcTensor);
```
