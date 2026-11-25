# CopyL1ToL0B

> [Code Location](../../../../../../../../include/catlass/conv/tile/copy_l1_to_l0b.hpp)

[TOC]

## Function

`CopyL1ToL0B` implements filter (convolution kernel) data movement from L1 to L0B (non-TLA style) in convolution scenarios. Using `LoadData2D`, it converts the `CI1KHKWCOCI0` layout to the `nZ` fractal format.

- Applicability: Atlas A2
- Style: non-TLA

## Template Prototype

```cpp
template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `L1Type` | `Gemm::GemmType<Element, CI1KHKWCOCI0>` |
| `L0Type` | L0B type. The default value is `void`.|

## Partial Specialization Implementation

| Partial Specialization| L1Type | LayoutSrc → LayoutDst| Description|
| :------ | :------ | :------ | :------ |
| A2 | `GemmType<Element, CI1KHKWCOCI0>` | CI1KHKWCOCI0 → nZ| LoadData 2D, KhKw-wise movement|
