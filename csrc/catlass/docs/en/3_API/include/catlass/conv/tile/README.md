# Conv/Tile Template Overview

> [Code Location](../../../../../../../include/catlass/conv/tile/)

[TOC]

Conv's tile-layer API serves as the template parameter for the Conv block layer. It orchestrates data movement and im2col operations in convolution scenarios. The API integrates GM → L1 movement, L1 → L0A (with im2col), L1 → L0B, L0C → GM (with type conversion and ReLU), aggregated templates, and other components.

## API List

| Component| Style| Applicable Hardware| Description|
| :------ | :------ | :------ | :------ |
| [copy_gm_to_l1](./copy_gm_to_l1/README.md) | Non-TLA + TLA| Atlas A2 and Ascend 950| GM → L1 movement (NC1HWC0/CI1KHKWCOCI0)|
| [copy_l1_to_l0a](./copy_l1_to_l0a/README.md) | Non-TLA + TLA| Atlas A2 and Ascend 950| L1 → L0A movement (NC1HWC0 → zZ, including im2col)|
| [copy_l1_to_l0b](./copy_l1_to_l0b/README.md) | Non-TLA + TLA| Atlas A2 and Ascend 950| L1 → L0B movement (CI1KHKWCOCI0 → nZ)|
| [copy_l0c_to_gm](./copy_l0c_to_gm/README.md) | Non-TLA + TLA| Atlas A2 and Ascend 950| L0C → GM write-back (zN → NC1HWC0, Fixpipe)|
| [tile_copy](./tile_copy/README.md) | Non-TLA + TLA| Atlas A2 and Ascend 950| Aggregation template for data movement (TileCopy/PackedTileCopyTla)|
