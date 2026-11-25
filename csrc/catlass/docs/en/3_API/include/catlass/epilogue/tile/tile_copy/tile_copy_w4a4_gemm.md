# TileCopyW4A4Gemm

> [Code Location](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyW4A4Gemm` is a template for epilogue W4A4 GEMM dequantization movement and aggregation. Similar to `TileCopyPerTokenDequant`, but it does not include per-channel scale movement (since W4A4 uses per-token scale and group size).

- Applicability: Atlas A2

## Template Prototype

```cpp
template <
    class ArchTag,
    class CType,                // int32_t accumulation result
    class PerTokenScaleType,    // Per-token scale (ColumnMajor)
    class DType                 // Destination type
>
struct TileCopyW4A4Gemm;
```

## Member Types

| Member Type| Description|
| :------ | :------ |
| `CopyGmToUbC` | `CopyGm2Ub<Arch, CType>` |
| `CopyGmToUbPerTokenScale` | `CopyGm2Ub<Arch, PerTokenScaleType>` |
| `CopyUbToGmD` | `CopyUb2Gm<Arch, DType>` |
