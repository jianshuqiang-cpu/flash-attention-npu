# TileCopyTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopyTla` is the core data movement template in the Tensor Layout Abstraction (TLA) style. It leverages SFINAE to automatically dispatch to the corresponding architecture-specific implementation. The two template parameters are the types of the source Tensor and the destination Tensor. Depending on the combination of layout and position, the appropriate partial specialization is automatically matched.

Unlike [TileCopyTlaExt](./tile_copy_tla_ext.md), `TileCopyTla` is fully dispatched automatically via SFINAE traits (such as `isRowMajor` and `iszN`), without the need for manually specifying LayoutTag.

## Primary Template Declaration

```cpp
template <
    class ArchTag,           // Architecture tag
    class TensorSrc,         // Source tensor type
    class TensorDst,         // Destination tensor type
    class Enable = void      // SFINAE enable
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyTla, can not find the specialization.");
};
```

## Partial Specialization Implementation

### Atlas A2 (Arch::AtlasA2, CATLASS_ARCH == 2201)

| Direction| SFINAE Condition| Implementation Location| API Reference|
| :------ | :------ | :------ | :------ |
| GM→L1 (RowMajor) | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md) |
| GM→L1 (ColumnMajor→zZ) | `isColumnMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md) |
| GM→L1 (VectorLayout) | `isVector<LayoutSrc>` && `isVector<LayoutDst>` | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md) |
| L1→L0A (RowMajor→zZ) | `isRowMajor<LayoutSrc>` && `hasL0ALayout<LayoutDst>` | `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0A (zN→zZ) | `iszN<LayoutSrc>` && `hasL0ALayout<LayoutDst>` | `atlasa2/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0B (ColumnMajor→nZ) | `isColumnMajor<LayoutSrc>` && `hasL0BLayout<LayoutDst>` | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| L1→L0B (zN→nZ) | `iszN<LayoutSrc>` && `hasL0BLayout<LayoutDst>` | `atlasa2/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| GM→UB | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `atlasa2/copy_gm_to_ub.hpp` | [copy_gm_to_ub](../copy_gm_to_ub/tile_copy_tla.md) |
| UB→GM | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `atlasa2/copy_ub_to_gm.hpp` | [copy_ub_to_gm](../copy_ub_to_gm/tile_copy_tla.md) |

### Ascend 950 (Arch::Ascend950, CATLASS_ARCH == 3510)

| Direction| SFINAE Condition| Implementation Location| API Reference|
| :------ | :------ | :------ | :------ |
| GM→L1 | `isRowMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `ascend950/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md) |
| GM→L1 (ColumnMajor) | `isColumnMajor<LayoutSrc>` && `isRowMajor<LayoutDst>` | `ascend950/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla.md) |
| L1→L0A (RowMajor→zZ) | `isRowMajor<LayoutSrc>` && `hasL0ALayout<LayoutDst>` | `ascend950/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0A (zN→zZ) | `iszN<LayoutSrc>` && `hasL0ALayout<LayoutDst>` | `ascend950/copy_l1_to_l0a.hpp` | [copy_l1_to_l0a](../copy_l1_to_l0a/tile_copy_tla.md) |
| L1→L0B (ColumnMajor→nZ) | `isColumnMajor<LayoutSrc>` && `hasL0BLayout<LayoutDst>` | `ascend950/copy_l1_to_l0b.hpp` | [copy_l1_to_l0b](../copy_l1_to_l0b/tile_copy_tla.md) |
| L1→BT | `isVector<LayoutSrc>` && `isVector<LayoutDst>` | `ascend950/copy_l1_to_bt.hpp` | [copy_l1_to_bt](../copy_l1_to_bt/tile_copy_tla.md) |

## APIs

All partial specializations provide a unified `operator()` interface.

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor
    TensorSrc const &srcTensor     // Source tensor
);
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"  // Include partial specializations.
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

auto gmLayout = tla::MakeLayout<Element, layout::RowMajor>(M, K);
auto l1Layout = tla::MakeLayout<Element, layout::RowMajor>(M, K);
auto gmTensor = tla::MakeTensor(gmData, gmLayout, Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data, l1Layout, Arch::PositionL1{});

// Automatically matched via SFINAE: GM RowMajor to L1 RowMajor
TileCopyTla<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor)> copyOp;
copyOp(l1Tensor, gmTensor);
```
