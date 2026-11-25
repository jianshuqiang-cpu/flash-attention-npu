# CopyGmToL1

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## Function

`CopyGmToL1` implements weight/Fmap data movement from global memory to L1 in convolution scenarios (non-TLA style). It supports two types of movement: Fmap (feature maps) and filter (convolution kernels).

- Applicability: Atlas A2
- Style: non-TLA

## Template Prototype

```cpp
template <class ArchTag, class GmType, class L1Type = void>
struct CopyGmToL1;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `GmType` | `Gemm::GemmType<Element, LayoutTag>`, including Element and Layout|
| `L1Type` | L1 data type. The default value is `void` (automatic inference).|

## Partial Specialization Implementation

| Partial Specialization| GmType | Description|
| :------ | :------ | :------ |
| Fmap-A | `GemmType<Element, NC1HWC0>` | NC1HWC0 → NC1HWC0, Cin1-wise movement|
| Filter-B | `GemmType<Element, CI1KHKWCOCI0>` | CI1KHKWCOCI0 → CI1KHKWCOCI0, including Cout alignment|

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // L1 destination
    AscendC::GlobalTensor<Element> srcTensor,   // Global memory source
    LayoutDst const &layoutDst,
    LayoutSrc const &layoutSrc
)
```

## Examples

### Fmap (NC1HWC0)

```cpp
#include "catlass/conv/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Conv::Tile;

constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;

using Element = half;
using LayoutTagSrc = layout::NC1HWC0;
using LayoutTagDst = layout::NC1HWC0;

using GmType = Gemm::GemmType<Element, LayoutTagSrc>;

auto layoutSrc = LayoutTagSrc::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);
auto layoutDst = LayoutTagDst::MakeLayout<Element>(1, Cin1, Hi, Wi, C0);

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = CopyGmToL1<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
