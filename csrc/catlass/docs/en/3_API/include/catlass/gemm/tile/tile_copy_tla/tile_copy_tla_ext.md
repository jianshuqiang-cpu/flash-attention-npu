# TileCopyTlaExt

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_copy_tla.hpp)

[TOC]

## Function

`TileCopyTlaExt` is an **extended variant** of `TileCopyTla` and additionally accepts two LayoutTag parameters: `LayoutTagSrc` and `LayoutTagDst`. This allows you to manually control the dispatch of partial specializations based on these two parameters, rather than relying on the layout traits embedded within the tensor.

When the source tensor layout is `PaddingRowMajor` or `PaddingColumnMajor` (not natively supported by `tla::MakeLayout`), you can explicitly specify `LayoutTagSrc = PaddingRowMajor` to dispatch to the corresponding partial specialization.

Differences from [TileCopyTla](./tile_copy_tla.md)

| Feature| TileCopyTla | TileCopyTlaExt |
| :------ | :------ | :------ |
| Dispatch method| SFINAE traits (such as `isRowMajor`)| Explicit LayoutTag template parameters|
| LayoutTag parameters| None (automatic deduction)| `LayoutTagSrc`, `LayoutTagDst` |
| Padding layout | Not supported| Supports via `PaddingRowMajor` or `PaddingColumnMajor`.|

## Primary Template Declaration

```cpp
template <
    class ArchTag,           // Architecture tag
    class TensorSrc,         // Source tensor type
    class TensorDst,         // Destination tensor type
    class LayoutTagSrc,      // Source LayoutTag (used for partial specialization matching)
    class LayoutTagDst       // Destination LayoutTag (used for partial specialization matching)
>
struct TileCopyTlaExt {
    static_assert(DEPENDENT_FALSE<ArchTag>,
        "Unsupported TileCopyTlaExt, can not find the specialization.");
};
```

> **Note**: `LayoutTagSrc` does not necessarily match the physical layout of the tensor (for example, you may specify `PaddingRowMajor` while the tensor is actually stored as `RowMajor`). It is used solely as a dispatch tag for partial specialization matching.

## Partial Specialization Implementation (Atlas A2)

| LayoutTagSrc | LayoutTagDst | Source Location| Destination Location| Implementation Location| API Reference|
| :------ | :------ | :------ | :------ | :------ | :------ |
| RowMajor / PaddingRowMajor | RowMajor | GM | L1 A1 | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla_ext.md) |
| ColumnMajor / PaddingColumnMajor | RowMajor | GM | L1 A1 | `atlasa2/copy_gm_to_l1.hpp` | [copy_gm_to_l1](../copy_gm_to_l1/tile_copy_tla_ext.md) |
| RowMajor | PaddingRowMajor | UB VECCALC | GM | `atlasa2/copy_ub_to_gm.hpp` | [copy_ub_to_gm](../copy_ub_to_gm/tile_copy_tla_ext.md) |

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor
);
```

## Examples

```cpp
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using Element = half;

// Matrix A PaddingRowMajor (after alignment)
auto gmLayout = tla::MakeLayout<Element, layout::PaddingRowMajor>(M, K_padded);
auto l1Layout = tla::MakeLayout<Element, layout::RowMajor>(M, K_padded);
auto gmTensor = tla::MakeTensor(gmData, gmLayout, Arch::PositionGM{});
auto l1Tensor = tla::MakeTensor(l1Data, l1Layout, Arch::PositionL1{});

// Explicit dispatch: PaddingRowMajor → RowMajor
TileCopyTlaExt<Arch::AtlasA2, decltype(gmTensor), decltype(l1Tensor),
    layout::PaddingRowMajor, layout::RowMajor> copyOp;
copyOp(l1Tensor, gmTensor);
```
