# GEMM Tile Class Template Overview

The tile-level API of GEMM serves as a template parameter for [blockMmad](../block/block_mmad.md) and typically does not need to be specified explicitly, as `BlockMmad` provides default configurations. Explicit declarations are required only during kernel template assembly when optimizing performance for specific scenarios or implementing dedicated features.

## API List

| Component                        | Description|
| :----------------------------------------------------------- | :------: |
| [tile_copy](./tile_copy/README.md)     |   A collection of all tile-level data transfer templates required for MMAD computations |
| [tile_mmad](./tile_mmad/README.md)     |   Tile-level MMAD computation |
| [tile_muls](./tile_muls.md)     |   Tile-level scalar multiplication |
| [tile_traits](./tile_traits.md)     |   Prologue trait wrapping |
| [tile_copy_tla](./tile_copy_tla/README.md)     |   TLA copy template base class declaration and implementation index |
| [copy_gm_to_l1](./copy_gm_to_l1/README.md)     |   Copy tile from GM to L1 |
| [copy_l1_to_l0a](./copy_l1_to_l0a/README.md)     |   Copy A matrix tile from L1 to L0A |
| [copy_l1_to_l0b](./copy_l1_to_l0b/README.md)     |   Copy B matrix tile from L1 to L0B |
| [copy_l1_to_bt](./copy_l1_to_bt/README.md)     |   Copy Bias Table from L1 to BT |
| [copy_l1_to_fp](./copy_l1_to_fp.md)     |   Copy data from L1 to GM through the FixPipe channel |
| [copy_l0c_to_dst](./copy_l0c_to_dst/README.md)     |   L0C copy shared infrastructure (quantization modes and enumeration definitions) |
| [copy_l0c_to_gm](./copy_l0c_to_gm/README.md)     |   Copy L0C accumulated result to GM |
| [copy_l0c_to_ub](./copy_l0c_to_ub.md)     |   Copy L0C accumulated result to UB |
| [copy_gm_to_ub](./copy_gm_to_ub/README.md)     |   Copy data from GM to UB |
| [copy_ub_to_gm](./copy_ub_to_gm/README.md)     |   Copy data from UB to GM |
| [cast_fp8_to_fp16](./cast_fp8_to_fp16.md)     |   FP8 dequantization and conversion to FP16 |
| [cast_int4_to_int8](./cast_int4_to_int8.md)     |   INT4 to INT8 conversion |
| [cast_int8_to_fp16](./cast_int8_to_fp16.md)     |   INT8 dequantization and conversion to FP16 |
