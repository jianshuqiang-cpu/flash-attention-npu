# copy_gm_to_l1

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_gm_to_l1.hpp)

[TOC]

## Overview

The `copy_gm_to_l1` module implements weight/feature map data movement from global memory to L1 in convolution scenarios. It supports both feature maps (Fmap, NC1HWC0) and filters (convolution kernel, CI1KHKWCOCI0). Each type of movement has non-TLA and TLA style implementations.

## API List

| API | Style| Applicable Hardware| Moving Object| Description|
| :------ | :------ | :------ | :------ | :------ |
| [CopyGmToL1](./copy_gm_to_l1.md) | Non-TLA| Atlas A2| Fmap/Filter| Basic movement, including partial specialization distribution|
| [CopyGmToL1ATla](./copy_gm_to_l1_a_tla.md) | TLA | Atlas A2 and Ascend 950| Fmap (matrix A)| NC1HWC0 → NC1HWC0|
| [CopyGmToL1BTla](./copy_gm_to_l1_b_tla.md) | TLA | Atlas A2 and Ascend 950| Filter (matrix B)| CI1KHKWCOCI0 → CI1KHKWCOCI0|

## Examples

### CopyGmToL1 (non-TLA, Fmap)

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

### CopyGmToL1ATla (TLA, Fmap)

```cpp
auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionGM{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL1{});

CopyGmToL1ATla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
