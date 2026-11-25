# copy_l0c_to_gm

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## Overview

The `copy_l0c_to_gm` module implements the write-back of convolution accumulation results from L0C (zN format) to global memory (NC1HWC0 format). It uses `AscendC::Fixpipe` as a direct path to complete data movement, type conversion, and optional ReLU.

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL0CToGm](./copy_l0c_to_gm.md) | Non-TLA| Atlas A2| Fixpipe + F322F16/BF16 + ReLU |
| [CopyL0CToGmTla](./copy_l0c_to_gm_tla.md) | TLA | Atlas A2 and Ascend 950| TLA version|

## Examples

### CopyL0CToGmTla (TLA)

```cpp
#include "catlass/conv/tile/atlasa2/copy_l0c_to_gm.hpp"

using namespace Catlass::Conv::Tile;

using ElementSrc = float;
using ElementDst = half;
constexpr uint32_t Cout = 64, Ho = 14, Wo = 14, C0 = 16;

auto layoutSrc = tla::MakeLayout<ElementSrc, layout::zN>(Ho * Wo, Cout);
auto layoutDst = tla::MakeLayout<ElementDst, layout::NC1HWC0>(1, Cout / C0, Ho, Wo, C0);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```
