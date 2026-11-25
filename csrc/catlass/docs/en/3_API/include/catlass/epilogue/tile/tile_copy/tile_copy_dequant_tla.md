# TileCopyDequantTla

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyDequantTla` is a template for epilogue TLA-style dequantization movement and aggregation. It references `CopyGm2UbTla` and `CopyUb2GmTla` as child components, each of which is a template requiring `TensorSrc`/`TensorDst` type parameterization for delayed instantiation.

- Applicability: Atlas A2 and Ascend 950
- Style: TLA

## Template Prototype

```cpp
template <
    class ArchTag,
    class ElementC_,   class LayoutTagC_,
    class ElementX_,   class LayoutTagX_,
    class ElementY_,   class LayoutTagY_,
    class ElementD_,   class LayoutTagD_
>
struct TileCopyDequantTla;
```

## Member Types

| Member Type| Description|
| :------ | :------ |
| `CopyGmToUbC` (template)| `CopyGm2UbTla<Arch, TensorC, TensorUbC>` |
| `CopyGmToUbX` (template)| `CopyGm2UbTla<Arch, TensorX, TensorUbX>` |
| `CopyGmToUbY` (template)| `CopyGm2UbTla<Arch, TensorY, TensorUbY>` |
| `CopyUbToGmD` (template)| `CopyUb2GmTla<Arch, TensorUbD, TensorD>` |
