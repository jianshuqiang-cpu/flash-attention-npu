# CopyL1ToL0A

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0a.hpp)

[TOC]

## Function

`CopyL1ToL0A` implements Fmap data movement from L1 to L0A (non-TLA style) in convolution  scenarios, while performing the im2col operation. Using `LoadData3D`/`LoadData3DParamsV2`, it converts the NC1HWC0 layout to the zZ fractal format.

- Applicability: Atlas A2
- Style: non-TLA

## Template Prototype

```cpp
template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `L1Type` | `Gemm::GemmType<Element, NC1HWC0>` |
| `L0Type` | L0A type. The default value is `void`.|

## Partial Specialization Implementation

| Partial Specialization| L1Type | LayoutSrc → LayoutDst| Description|
| :------ | :------ | :------ | :------ |
| A2 | `GemmType<Element, NC1HWC0>` | NC1HWC0 → zZ| LoadData 3D v2, including im2col|

The constructor receives a `Conv2dFilterParams` parameter.

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // (ho, wo, cin1, Kh, Kw, C0) zZ format
    AscendC::LocalTensor<Element> srcTensor,    // (cin1, hi, wi, C0) NC1HWC0 format
    LayoutDst const &layoutDst,                 // zZ layout
    LayoutSrc const &layoutSrc,                 // NC1HWC0 layout
    uint8_t *blockPadList                       // {padLeft, padRight, padTop, padBottom}
)
```
