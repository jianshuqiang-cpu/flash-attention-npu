# TileCopy (Data Movement Template Collection)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## Overview

`TileCopy` is a **data movement template collection** at the GEMM Tile layer. The types of all data movement subcomponents, such as GM → L1, L1 → L0A/B, L0C → GM(Dst/UB), Bias, and Scale, are deduced based on template parameters for selection during blockMmad assembly.

All templates provide only `using` type aliases and do not perform any actual computation.

## API List

### Non-TLA style

| Template| Applicable Hardware| Inheritance Relationship| Description|
| :------ | :------ | :------ | :------ |
| [TileCopy](./tile_copy.md) | Atlas A2 + Ascend 950| Base class| Basic data movement template collection|
| [TileCopyWithPrologueDeqPerTensor](./tile_copy_prologue_deq_per_tensor.md) | Atlas A2| — | + Prologue + per-tensor dequantization|
| [TileCopyWithPrologue](./tile_copy_prologue.md) | Atlas A2| — | + Prologue preprocessing|
| [TileCopyGemm](./tile_copy_gemm.md) | Atlas A2 + Ascend 950| — | GEMM-specific selector|
| [ConvTileCopy](./conv_tile_copy.md) | Atlas A2 + Ascend 950| — | Conv-specific|
| [ReluTileCopy](./relu_tile_copy.md) | Atlas A2 + Ascend 950| Inherits from TileCopy.| +ReLU write-back|
| [QuantTileCopy](./quant_tile_copy.md) | Atlas A2| Inherits from TileCopy.| +Scale/FP channels + on-the-fly quantization|

### TLA Style

| Template| Applicable Hardware| Inheritance Relationship| Description|
| :------ | :------ | :------ | :------ |
| [SparseTileCopyTla](./sparse_tile_copy_tla.md) | Atlas A2| — | Sparse GEMM (TLA)|
| [PackedTileCopyTla](./packed_tile_copy_tla.md) | Atlas A2 + Ascend 950| — | Core TLA collection|
| [PaddingPackedTileCopyTla](./padding_packed_tile_copy_tla.md) | Atlas A2 + Ascend 950| — | +Padding support|
| [PackedTileCopyTlaToUB](./packed_tile_copy_tla_to_ub.md) | Ascend 950| Inherits from PackedTileCopyTla.| + UB destination|
| [PackedMxTileCopyTla](./packed_mx_tile_copy_tla.md) | Atlas A2| Inherits from PackedTileCopyTla.| +MX Scale channels|
| [PackedMxA8W4TileCopyTla](./packed_mx_a8w4_tile_copy_tla.md) | Atlas A2| Inherits from PackedTileCopyTla.| +MX Scale + A8W4 |

## Template Inheritance Relationship

```
TileCopy
├── ReluTileCopy            (overrides CopyL0CToGm → ReLU)
├── QuantTileCopy           (overrides CopyL0CToGm + adds Scale/FP channels)
│
PackedTileCopyTla
├── PackedTileCopyTlaToUB   (overrides CopyL0CToDst → UB)
├── PackedMxTileCopyTla     (adds MxScale channels)
└── PackedMxA8W4TileCopyTla (adds MxScale + overrides L1 → L0B)
```

## Template Selection Guide

| Scenario| Recommended Template| Style| Architecture|
| :------ | :------ | :------ | :------ |
| GEMM (FP16/ BF16)| `TileCopy`/`PackedTileCopyTla`| Non-TLA/TLA| Not limited|
| GEMM with explicit L1/L0 derivation| `TileCopyGemm` | Non-TLA| Not limited|
| INT8 quantized inference| `QuantTileCopy` | Non-TLA| Atlas A2|
| Prologue dequantization| `TileCopyWithPrologueDeqPerTensor` | Non-TLA| Atlas A2|
| Prologue preprocessing (INT4 → INT8)| `TileCopyWithPrologue` | Non-TLA| Atlas A2|
| ReLU activation| `ReluTileCopy` | Non-TLA| Not limited|
| Conv GEMM (Im2Col)| `ConvTileCopy` | Non-TLA| Not limited|
| Sparse GEMM| `SparseTileCopyTla` | TLA | Atlas A2|
| Padding Block GEMM | `PaddingPackedTileCopyTla` | TLA | Not limited|
| L0C → UB (Ascend 950)| `PackedTileCopyTlaToUB` | TLA | Ascend 950|
| FP8 MX Scale | `PackedMxTileCopyTla` | TLA | Atlas A2|
| FP8 MX Scale + INT4 weight | `PackedMxA8W4TileCopyTla` | TLA | Atlas A2|
