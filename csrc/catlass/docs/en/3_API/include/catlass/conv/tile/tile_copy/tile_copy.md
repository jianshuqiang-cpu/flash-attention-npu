# TileCopy (Conv)

> [Code Location](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopy` is the non-TLA version (Atlas A2) for convolution movement and aggregation. It combines and references four child movement components, exposing them as type members for use by block-level convolution.

- Applicability: Atlas A2
- Style: non-TLA

## Template Prototype

```cpp
template <class ArchTag, class FmapType, class FilterType, class OutputType, class BiasType = void>
struct TileCopy;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `FmapType` | GemmType of Fmap|
| `FilterType` | GemmType of filter|
| `OutputType` | GemmType of Output|
| `BiasType` | Bias type. The default value is `void`.|

## Member Types

| Member Type| Child Component| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `Conv::Tile::CopyGmToL1<Arch, FmapType>` | Fmap, GM → L1|
| `CopyGmToL1B` | `Conv::Tile::CopyGmToL1<Arch, FilterType>` | Filter, GM → L1|
| `CopyL1ToL0A` | `Conv::Tile::CopyL1ToL0A<...>` | Fmap, L1 → L0A (im2col)|
| `CopyL1ToL0B` | `Conv::Tile::CopyL1ToL0B<...>` | Filter, L1 → L0B|
| `CopyL0CToGm` | `Conv::Tile::CopyL0CToGm<...>` | L0C → GM (including Fixpipe/F322F16)|
