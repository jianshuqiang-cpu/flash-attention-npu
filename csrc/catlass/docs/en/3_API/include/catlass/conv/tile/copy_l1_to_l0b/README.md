# copy_l1_to_l0b

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## Overview

The `copy_l1_to_l0b` module implements filter (convolution kernel) data movement from L1 to L0B in the convolution scenario (CI1KHKWCOCI0 → nZ).

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [CopyL1ToL0B](./copy_l1_to_l0b.md) | Non-TLA| Atlas A2| LoadData 2D |
| [CopyL1ToL0BTla](./copy_l1_to_l0b_tla.md) | TLA | Atlas A2 and Ascend 950| TLA version|

## Examples

### CopyL1ToL0BTla (TLA)

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0b.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Kh = 3, Kw = 3, Cout = 64, C0 = 16;

auto layoutSrc = tla::MakeLayout<Element, layout::CI1KHKWCOCI0>(Cin1, Kh, Kw, Cout, C0);
auto layoutDst = tla::MakeLayout<Element, layout::nZ>(Cin1 * Kh * Kw, Cout);

auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0B{});

CopyL1ToL0BTla<Element> copyOp;
copyOp(dstTensor, srcTensor);
```
