# copy_l1_to_l0a

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## Overview

The `copy_l1_to_l0a` module implements Fmap data movement from L1 to L0A in convolution scenarios, while performing the im2col operation (NC1HWC0 → zZ).

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL1ToL0A](./copy_l1_to_l0a.md) | Non-TLA| Atlas A2| LoadData 3D v2, including im2col|
| [CopyL1ToL0ATla](./copy_l1_to_l0a_tla.md) | TLA | Atlas A2 and Ascend 950| `LoadDataWithStride` added to 950|

## Examples

### CopyL1ToL0ATla (TLA)

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0a.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;
constexpr uint32_t Kh = 3, Kw = 3;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::zZ>(16, 27);

Conv2dFilterParams params{.strideW_ = 1, .strideH_ = 1, .kw_ = Kw, .kh_ = Kh, .dilationW_ = 1, .dilationH_ = 1};

uint8_t padList[4] = {0, 0, 0, 0};

CopyL1ToL0ATla<Element> copyOp(params);
copyOp(dstTensor, srcTensor, padList);
```
