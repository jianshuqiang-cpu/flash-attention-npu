# CopyL1ToL0ATla

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## Function

`CopyL1ToL0ATla` implements Fmap data movement from L1 to L0A (TLA style) in convolution scenarios, while performing the im2col operation.

- Atlas A2: `LoadData` and `Conv2dFilterParams`
- Ascend 950: `LoadDataWithStride` and `SetLoadDataRepeatWithStride`

## Template Prototype

```cpp
template <class Element>
struct CopyL1ToL0ATla;
```

| Parameter| Description|
| :------ | :------ |
| `Element` | Element type, for example, `half`|

The constructor receives a `Conv2dFilterParams` parameter.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint8_t *blockPadList
)
```

## Examples

```cpp
#include "catlass/conv/tile/atlasa2/copy_l1_to_l0a.hpp"

using namespace Catlass::Conv::Tile;

using Element = half;
constexpr uint32_t Cin1 = 4, Hi = 28, Wi = 28, C0 = 16;
constexpr uint32_t Kh = 3, Kw = 3;

auto layoutSrc = tla::MakeLayout<Element, layout::NC1HWC0>(1, Cin1, Hi, Wi, C0);
auto layoutDst = tla::MakeLayout<Element, layout::zZ>(16, 27);

Conv2dFilterParams params{.strideW_ = 1, .strideH_ = 1, .kw_ = Kw, .kh_ = Kh, .dilationW_ = 1, .dilationH_ = 1};

AscendC::LocalTensor<Element> srcData, dstData;
auto srcTensor = tla::MakeTensor(srcData, layoutSrc, Arch::PositionL1{});
auto dstTensor = tla::MakeTensor(dstData, layoutDst, Arch::PositionL0A{});

uint8_t padList[4] = {0, 0, 0, 0};

CopyL1ToL0ATla<Element> copyOp(params);
copyOp(dstTensor, srcTensor, padList);
```
