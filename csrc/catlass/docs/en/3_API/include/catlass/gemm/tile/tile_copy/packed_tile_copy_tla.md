# PackedTileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Function

`PackedTileCopyTla` is the **core** Tensor Layout Abstraction (TLA) data movement template collection at the GEMM Tile layer. All movement operators are of type `TileCopyTla` or `CopyL0CToGmTla`, and are driven by LayoutTag through a complete layout derivation chain: GM LayoutTag → L1 LayoutTag → L0 LayoutTag → tla::Layout → tla::Tensor.

The template supports features, such as ReLU, dequantization (per-tensor/per-channel), and bias, all controlled via template parameter switches.

## Referenced Tile Components

| Member Alias| Referenced Underlying Template| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `TileCopyTla<ArchTag, TensorA, TensorL1A>` | Matrix A: GM → L1 (TLA)|
| `CopyGmToL1B` | `TileCopyTla<ArchTag, TensorB, TensorL1B>` | Matrix B: GM → L1 (TLA)|
| `CopyGmToL1Bias` | `TileCopyTla<ArchTag, TensorBias, TensorL1Bias>` or `EmptyClass`| Bias: GM → L1 (conditional)|
| `CopyGmToL1Scale` | `TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>` or `EmptyClass`| Scale: GM → L1 (per-channel)|
| `CopyL1ToL0A` | `TileCopyTla<ArchTag, TensorL1A, TensorL0A>` | Matrix A: L1 → L0A (TLA)|
| `CopyL1ToL0B` | `TileCopyTla<ArchTag, TensorL1B, TensorL0B>` | Matrix B: L1 → L0B (TLA)|
| `CopyL1ToBT` | `TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>` or `EmptyClass`| Bias: L1 → BT (conditional)|
| `CopyL0CToDst` (Ascend 950) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>` | L0C → Dst (TLA)|
| `CopyL0CToGm` (Atlas A2) | `CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>` | L0C → GM (TLA)|

## Template Prototype

```cpp
template <
    class ArchTag,                                                   // Architecture tag
    class ElementA_,                                                 // A matrix element type
    class LayoutTagA_,                                               // GM layout tag of matrix A
    class ElementB_,                                                 // B matrix element type
    class LayoutTagB_,                                               // GM layout tag of matrix B
    class ElementC_,                                                 // C matrix element type
    class LayoutTagC_,                                               // GM layout tag of matrix C
    class ElementBias = void,                                        // (Optional) Bias element type
    bool ReluEnable_ = false,                                        // ReLU switch
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT, // Dequantization granularity
    class L0CCopyMode = CopyToGM                                     // L0C→Dst transfer mode
>
struct PackedTileCopyTla;
```

## Template Parameters

| Parameter| Default Value| Description|
| :------ | :------ | :------ |
| `ElementBias` | `void` | If this parameter is not set to **void**, the bias data movement channel is enabled.|
| `ReluEnable_` | `false` | ReLU enable flag, passed to `CopyL0CToGmTla`|
| `DEQUANT_GRANULARITY_` | `NO_QUANT` | `PER_TENSOR` / `PER_CHANNEL` / `NO_QUANT` |
| `L0CCopyMode` | `CopyToGM` | `CopyToGM` for Atlas A2, `CopyToUB` for Ascend 950|

## Layout Derivation Chain (Using RowMajor A as an Example)

```
LayoutTagA_ = RowMajor
  → L1ATypeSelector → LayoutTagL1A = v2 (zN)
    → TagToLayout_t → LayoutL1A = tla::Layout<Shape<M,K>, Stride<...>>
      → TensorL1A = tla::Tensor<LocalTensor<half>, LayoutL1A, Coord<0,0>, A1>
  → L0ALayoutSelector → LayoutTagL0A = zZ
    → TagToLayout_t → LayoutL0A = tla::Layout<Shape<...>, Stride<...>>
      → TensorL0A = tla::Tensor<LocalTensor<half>, LayoutL0A, Coord<0,0>, A2>
```

## Examples

### Basic Call (No Bias, No ReLU, and No Quantization)

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::PackedTileCopyTla<
    Arch::AtlasA2,
    half, layout::RowMajor,
    half, layout::ColumnMajor,
    half, layout::RowMajor>;

typename TileCopy_::CopyGmToL1A copyGmToL1A;
typename TileCopy_::CopyL1ToL0A copyL1ToL0A;
typename TileCopy_::CopyL0CToGm copyL0CToGm;
```

### Complete Call (Bias, ReLU, and Per-Tensor Quantization)

```cpp
using TileCopy_ = Tile::PackedTileCopyTla<
    Arch::AtlasA2,
    int8_t, layout::RowMajor,
    int8_t, layout::ColumnMajor,
    int8_t, layout::RowMajor,
    half,                                    // ElementBias
    false,                                    // ReluEnable
    ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1Bias  copyGmToL1Bias;
typename TileCopy_::CopyL1ToBT      copyL1ToBT;
typename TileCopy_::CopyL0CToGm copyL0CToGm; // Per-tensor quantized write-back
```
