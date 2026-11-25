# Tile Copy Basic Template
>
> [Code location](../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

## Description

Integrates all partial specialization branches and data types for data transfer paths across the hardware memory hierarchy at the tile level. It primarily includes:

- `ElementA`: data type of the left matrix
- `ElementB`: data type of the right matrix
- `ElementAccumulator`: data type of the accumulated data in L0C
- [`CopyGmToL1A`](./copy_gm_to_l1.md): partial specialization implementation for transferring matrix A tile blocks from GM to L1
- `CopyGmToL1B`: partial specialization implementation for transferring matrix B tile blocks from GM to L1
- `CopyL1ToL0A`: partial specialization implementation for transferring matrix A tile blocks from L1 to L0A
- `CopyL1ToL0B`: partial specialization implementation for transferring matrix B tile blocks from L1 to L0B
- `CopyL0CToGm`: partial specialization implementation for transferring result matrix tile blocks from L0C to GM
- `CopyGmToL1Bias`: partial specialization implementation for transferring bias matrix tile blocks from GM to L1
- `CopyL1ToBT`: partial specialization implementation for transferring bias matrix tile blocks from L1 to BT

## Template List

### TileCopy

- Function description:
- Template parameters:
- Example:

### TileCopyWithPrologueDeqPerTensor

- Function description:
- Template parameters:
- Example:
