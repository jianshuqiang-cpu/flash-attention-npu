# TileBroadcastOneBlk

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## Function

`TileBroadcastOneBlk` implements the one-block broadcast operation in the epilogue stage. It broadcasts a single element from Unified Buffer across an entire block (32 bytes), which is commonly used to broadcast scalar scale or zero-point values so they can participate in vector computations.

- Applicability: all architectures (no architecture specialization)
- Style: non-TLA

## Template Prototype

```cpp
template <
    class ArchTag_,           // Architecture tag
    class ComputeType_,       // Computation data type (including Element)
    uint32_t COMPUTE_LENGTH_  // Computation length
>
struct TileBroadcastOneBlk;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag_` | Architecture tag|
| `ComputeType_` | Computation data type. The element type is obtained through `ComputeType_::Element`.|
| `COMPUTE_LENGTH_` | Total number of elements to be broadcast|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,    // Destination Unified Buffer (after broadcast)
    AscendC::LocalTensor<ElementCompute> const &ubIn      // Source Unified Buffer (COMPUTE_LENGTH elements)
)
```

Internally, each element is broadcast to the entire block using `AscendC::Brcb` (BroadCast to Block) and `BrcbRepeatParams`.

## Examples

```cpp
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 256;

using BroadcastOp = TileBroadcastOneBlk<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> ubOut, ubIn;

BroadcastOp broadcastOp;
broadcastOp(ubOut, ubIn);
```
