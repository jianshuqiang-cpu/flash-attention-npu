# TileCopyTla Series (Base Template Class for TLA-based Data Movement)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Overview

`tile_copy_tla.hpp` declares the **base class definitions** for all Tensor Layout Abstraction (TLA) data movement templates at the CATLASS Tile layer, including `TileCopyTla`, `TileCopyTlaExt`, `TileCopySparseTla`, `CopyL1ToL0BSparseTla`, and `TileCopyFAQTla`. Specific implementations are located in the corresponding `copy_*.hpp` file in the `atlasa2/` and `ascend950/` subdirectories.

## API List

| Template| Dispatch Method| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [TileCopyTla](./tile_copy_tla.md) | SFINAE trait | Atlas A2 + Ascend 950| Core TLA data movement template with automatic layout matching|
| [TileCopyTlaExt](./tile_copy_tla_ext.md) | Explicit LayoutTag| Atlas A2| Extended version with support for Padding layouts|
| [TileCopySparseTla](./tile_copy_sparse_tla.md) | SFINAE trait | Atlas A2| Sparse GEMM data movement|
| [CopyL1ToL0BSparseTla](./copy_l1_to_l0b_sparse_tla.md) | SFINAE trait | Atlas A2| Sparse B matrix transfer: L1 → L0B (with index tensor)|
| [TileCopyFAQTla](./tile_copy_faq_tla.md) | Fixed match| Atlas A2| FA LoadQ GM→L1 zN |

## Template Relationship Diagram

```
tile_copy_tla.hpp
├── TileCopyTla           → 9 partial specializations (GM → L1, L1 → L0A/B, GM → UB, UB → GM, L1 → BT)
├── TileCopyTlaExt        → 3 partial specializations (PaddingRowMajor, PaddingColumnMajor)
├── TileCopySparseTla     → 3 partial specializations (GM → L1A, GM → L1B, L1 → L0A)
├── CopyL1ToL0BSparseTla  → 1 partial specialization (L1 → L0B + index)
└── TileCopyFAQTla        → 1 partial specialization (GM RowMajor → L1 zN)
```

## Implementing Indexing by Module

The distribution of partial specialization implementations is as follows.

| Implementation File| Included Partial Specialization|
| :------ | :------ |
| `atlasa2/copy_gm_to_l1.hpp` | TileCopyTla×3, TileCopyTlaExt×2, TileCopySparseTla×2, TileCopyFAQTla×1 |
| `atlasa2/copy_gm_to_ub.hpp` | TileCopyTla×1 |
| `atlasa2/copy_l1_to_l0a.hpp` | TileCopyTla×2, TileCopySparseTla×1 |
| `atlasa2/copy_l1_to_l0b.hpp` | TileCopyTla×2, CopyL1ToL0BSparseTla×1 |
| `atlasa2/copy_ub_to_gm.hpp` | TileCopyTla×1, TileCopyTlaExt×1 |
| `ascend950/copy_gm_to_l1.hpp` | TileCopyTla×2 |
| `ascend950/copy_l1_to_l0a.hpp` | TileCopyTla×2 |
| `ascend950/copy_l1_to_l0b.hpp` | TileCopyTla×1 |
| `ascend950/copy_l1_to_bt.hpp` | TileCopyTla×1 |

## Template Selection Guide

| Scenario| Recommendation|
| :------ | :------ |
| Common data movement from GM to L1| `TileCopyTla` |
| Data movement from L1 to L0A/L0B| `TileCopyTla` |
| GM source: PaddingRowMajor/PaddingColumnMajor| `TileCopyTlaExt` |
| Sparse GEMM| `TileCopySparseTla` + `CopyL1ToL0BSparseTla` |
| FlashAttention Q matrix load| `TileCopyFAQTla` |

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"   // Include partial specializations.
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

// Construct the TLA tensor.
auto gmTensor = tla::MakeTensor(gmData,
    tla::MakeLayout<half, layout::RowMajor>(M, K), Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data,
    tla::MakeLayout<half, layout::RowMajor>(M, K), Arch::PositionL1{});

// SFINAE automatic dispatch
TileCopyTla<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor)> copyOp;
copyOp(l1Tensor, gmTensor);
```
