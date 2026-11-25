# TileCopyWithPrologueDeqPerTensor

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyWithPrologueDeqPerTensor` is a data movement template collection that includes Prologue dequantization (per-tensor) and extends [TileCopy](./tile_copy.md) by adding PrologueA and PrologueB operator types, while fixing `CopyL0CToGm` to perform per-tensor quantization.

This template is designed for quantized inference scenarios: quantized weights stored in Global Memory (GM), dequantization during the Prologue stage, and per-tensor quantization applied on the L0C-to-GM write-back.

> **Restriction**: Only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) is supported.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `CopyGmToL1<ArchTag, AType>` | Matrix A: GM → L1|
| `CopyGmToL1B` | `CopyGmToL1<ArchTag, BType>` | Matrix B: GM → L1|
| `CopyL1ToL0A` | `CopyL1ToL0A<ArchTag, L1AType>` | Matrix A: L1 → L0A|
| `CopyL1ToL0B` | `CopyL1ToL0B<ArchTag, L1BType>` | Matrix B: L1 → L0B|
| `CopyL0CToGm` | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, PER_TENSOR>` | L0C → GM (per-tensor)|
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>` or `void`| Bias: GM → L1|
| `CopyL1ToBT` | `CopyL1ToBT<ArchTag, ...>` or `void` | Bias: L1 → BT|

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
struct TileCopyWithPrologueDeqPerTensor;
```

## Member Type Derivation

```cpp
using ElementA = typename AType::Element;
using ElementB = typename BType::Element;
using ElementAccumulator =
    typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

using CopyGmToL1A = CopyGmToL1<ArchTag, AType>;
using CopyGmToL1B = CopyGmToL1<ArchTag, BType>;

using PrologueA = PrologueA_;
using PrologueB = PrologueB_;

using CopyL1ToL0A = CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
using CopyL1ToL0B = CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType, ScaleGranularity::PER_TENSOR>;
using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>, void, ...>;
using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>, void, ...>;
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
using BType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
using CType = Gemm::GemmType<int8_t, layout::RowMajor>;

using PrologueA = Tile::TileCastInt8ToFp16Dequant<Arch::AtlasA2, ...>;
using PrologueB = Tile::TileCastInt8ToFp16Dequant<Arch::AtlasA2, ...>;

using TileCopy_ = Tile::TileCopyWithPrologueDeqPerTensor<
    Arch::AtlasA2, AType, BType, CType, PrologueA, PrologueB>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;  // Per-tensor quantization
```
