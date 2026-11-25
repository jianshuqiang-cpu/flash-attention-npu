# tile_copy (Conv)

> [Code Location](../../../../../../../../include/catlass/conv/tile/tile_copy.hpp)

[TOC]

## Overview

Conv `tile_copy` is an aggregated movement template. It combines and references child movement components of GM → L1, L1 → L0A/L0B, and L0C → GM in the convolution scenario, exposing them as type members for use by block-level convolution.

## API List

| API | Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [TileCopy](./tile_copy.md) | Non-TLA| Atlas A2| Basic aggregation, referencing 4 child components|
| [PackedTileCopyTla](./packed_tile_copy_tla.md) | TLA | Atlas A2 and Ascend 950| Automatic inference of intermediate layouts, architecture-adaptive|

## Examples

### PackedTileCopyTla

```cpp
#include "catlass/conv/tile/tile_copy.hpp"

using namespace Catlass::Conv::Tile;

using ElementFmap = half;
using ElementFilter = half;
using ElementOutput = half;

using LayoutTagFmap = layout::NC1HWC0;
using LayoutTagFilter = layout::CI1KHKWCOCI0;
using LayoutTagOutput = layout::NC1HWC0;

using Copy = PackedTileCopyTla<Arch::AtlasA2,
    ElementFmap, LayoutTagFmap,
    ElementFilter, LayoutTagFilter,
    ElementOutput, LayoutTagOutput
>;
```
