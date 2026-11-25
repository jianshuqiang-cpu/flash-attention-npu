# TileCopyWithPrologue

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyWithPrologue` is a data movement template collection that extends [TileCopy](./tile_copy.md) by adding `PrologueA` and `PrologueB` operator types. Unlike [TileCopyWithPrologueDeqPerTensor](./tile_copy_prologue_deq_per_tensor.md), `CopyL0CToGm` retains the default behavior (`NO_QUANT`).

This template is intended for scenarios that require Prologue preprocessing (for example, type conversion from INT4 to INT8) but do not need on-the-fly quantization.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType>` | Matrix A: Global Memory (GM) → L1|
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType>` | Matrix B: GM → L1|
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType>` | Matrix A: L1 → L0A|
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType>` | Matrix B: L1 → L0B|
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, NO_QUANT>` | L0C → GM (non-quantized)|
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>` or `void`| Bias: GM → L1|
| `CopyL1ToBT` | `CopyL1ToBT<ArchTag, ...>` or `void`| Bias: L1 → BT|

## Template Prototype

```cpp
template <
    class ArchTag,            // Architecture tag: Arch::AtlasA2
    class AType,              // A matrix GmType
    class BType,              // B matrix GmType
    class CType,              // C matrix GmType
    class PrologueA_,         // Prologue operator type of matrix A
    class PrologueB_,         // Prologue operator type of matrix B
    class BiasType = void     // Bias GmType (optional)
>
struct TileCopyWithPrologue;
```

## Member Type Derivation

Mostly identical to `TileCopy`, with the following additions:
```cpp
using PrologueA = PrologueA_;
using PrologueB = PrologueB_;
```

## Differences from TileCopyWithPrologueDeqPerTensor

| Template| CopyL0CToGm Quantization Granularity| Scenario|
| :------ | :------ | :------ |
| `TileCopyWithPrologue` | `NO_QUANT` (default)| Prologue preprocessing, no quantization on write-back|
| `TileCopyWithPrologueDeqPerTensor` | `PER_TENSOR` | Prologue dequantization + per-tensor quantization on write-back|

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
using BType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::RowMajor>;

using PrologueB = Tile::TileCastInt4ToInt8<Arch::AtlasA2, ...>;

using TileCopy_ = Tile::TileCopyWithPrologue<
    Arch::AtlasA2, AType, BType, CType, PrologueA, PrologueB>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```
