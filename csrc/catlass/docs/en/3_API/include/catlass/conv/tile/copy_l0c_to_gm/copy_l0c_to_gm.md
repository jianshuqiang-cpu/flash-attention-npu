# CopyL0CToGm

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l0c_to_gm.hpp)

[TOC]

## Function

`CopyL0CToGm` implements the non-TLA version of writing convolution accumulation results from L0C (zN format) to global memory (NC1HWC0 format). It uses `AscendC::Fixpipe` as a direct path to complete data movement, type conversion (F322F16/F322BF16), and optional ReLU activation.

- Applicability: Atlas A2
- Style: non-TLA

## Template Prototype

```cpp
template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGm;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `ElementAccumulator` | Accumulation element type, for example, `float`|
| `GmType` | `Gemm::GemmType<ElementDst, NC1HWC0>` |
| `DEQUANT_GRANULARITY` | Quantization mode: `NO_QUANT`, `PER_TENSOR`, `PER_CHANNEL` or `PER_GROUP`|
| `ReluEnable` | Whether to enable ReLU. The default value is `false`.|

## Partial Specialization Implementation

| Partial Specialization| GmType | Description|
| :------ | :------ | :------ |
| A2, NO_QUANT | `GemmType<ElementDst, NC1HWC0>` | zN → NC1HWC0, Ho-wise Fixpipe|

## APIs

```cpp
void operator()(
    AscendC::GlobalTensor<ElementDst> const &dst,    // (Batch, Cout1, Ho, Wo, C0) NC1HWC0
    AscendC::LocalTensor<ElementSrc> const &src,     // L0C zN format
    LayoutDst const &dstLayout,
    uint8_t unitFlag = 0
)
```
