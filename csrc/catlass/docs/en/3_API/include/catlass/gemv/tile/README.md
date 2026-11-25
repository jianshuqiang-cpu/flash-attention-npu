# GEMV/Tile Template Overview

> [Code Location](../../../../../../../include/catlass/gemv/tile/)

[TOC]

The tile-level GEMV API serves as the template parameter for the GEMV block layer. It handles data movement and vector computation in matrix-vector multiplication scenarios. Depending on the chip type, it provides two data paths: AIV (bidirectional data movement between Global Memory and Unified Buffer) and AIC (data movement from Global Memory to L1, then to L0, and finally back to Global Memory).

## API List

### Data Movement Components

| Component| Applicable Hardware| Description|
| :------ | :------ | :------ |
| [vec_copy_gm_to_ub](./vec_copy_gm_to_ub.md) | All architectures| Vector-level data movement from Global Memory to Unified Buffer|
| [vec_copy_ub_to_gm](./vec_copy_ub_to_gm.md) | Atlas A2| Vector-level data movement from Unified Buffer to Global Memory (including atomic add mode)|
| [matrix_copy_gm_to_ub](./matrix_copy_gm_to_ub.md) | Atlas A2| Matrix-level data movement from Global Memory to Unified Buffer (RowMajor/ColumnMajor, three-level adaptive)|
| [tile_copy](./tile_copy/README.md) | Atlas A2, Ascend 950 | Aggregation template for data movement (TileCopyGemvAiv/TileCopyGemvAic)|

### Vector Computation

| Component| Applicable Hardware| Description|
| :------ | :------ | :------ |
| [tile_vmuls](./tile_vmuls.md) | All architectures| Vector-scalar multiplication (Muls)|
| [tile_vmad](./tile_vmad.md) | Atlas A2| Vector-matrix multiply-accumulate (Y += A * X), supporting RowMajor/ColumnMajor|
