# QuantTileCopy

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`QuantTileCopy` is inherited from [TileCopy](./tile_copy.md) and **overrides** `CopyL0CToGm` to support a specified quantization granularity (default `PER_TENSOR`). It also **adds** two new data movement channels: `CopyGmToL1Scale` and `CopyL1ToFP`, for loading the quantization Scale data and bypassing it to the FixPipe.

This template is designed for quantized inference or training scenarios. The Scale data stored in Global Memory (GM) is loaded into L1 via `CopyGmToL1Scale`. It is then bypassed to the FixPipe via `CopyL1ToFP`. During the L0C-to-GM write-back, the quantization is applied on-the-fly using the Scale data.

## Referenced Tile Components

| Member Alias| Source|
| :------ | :------ |
| `CopyGmToL1A` ~ `CopyL1ToBT` | Inherited from `TileCopy<ArchTag, AType, BType, CType, BiasType>`|
| `CopyL0CToGm` (overridden)| `CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>` |
| `CopyGmToL1Scale` (new)| `CopyGmToL1<ArchTag, uint64_t VectorLayout GM, uint64_t VectorLayout A1>` |
| `CopyL1ToFP` (new)| `CopyL1ToFP<ArchTag, uint64_t VectorLayout A1, uint64_t VectorLayout C2PIPE2GM>` |

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag
    class AType,                                                     // GmType of matrix A
    class BType,                                                     // GmType of matrix B
    class CType,                                                     // GmType of matrix C
    class BiasType = void,                                           // Bias GmType (optional)
    ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR      // Quantization granularity
>
struct QuantTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType>;
```

## Overridden and Newly Added Members

```cpp
// Overriding: quantized L0C → GM
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;

//New: Scale GM → L1
using CopyGmToL1Scale = CopyGmToL1<ArchTag,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>>;

//New: L1 → FP (FixPipe)
using CopyL1ToFP = CopyL1ToFP<ArchTag,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>>;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::QuantTileCopy<Arch::AtlasA2, AType, BType, CType,
    void, ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1Scale copyGmToL1Scale;   // Scale copy-in
typename TileCopy_::CopyL1ToFP      copyL1ToFP;         // Scale bypass
typename TileCopy_::CopyL0CToGm     copyL0CToGm;        // On-the-fly quantized write-back
```
