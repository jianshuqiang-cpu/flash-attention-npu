# GEMM Block Swizzle Class Template Overview
>
> [Code location](../../../../../../../../include/catlass/gemm/block/block_swizzle.hpp)

## Description

The swizzle policy determines the allocation relationship and computation sequence of the basic task blocks across the AI Cores. For a deeper breakdown of the underlying mechanics, see [Swizzle Policies](../../../../../../2_Design/01_kernel_design/02_swizzle.md).

## Common Methods

- Constructor: Calculates the number of block splits along the relevant dimensions based on parameters such as the actual problem shape and the tiling configuration.
- `void Update`: Updates the number of block splits along the relevant dimensions.
- `uint32_t GetCoreLoops`: Calculates and returns the total count of split basic blocks.
- `GemmCoord GetBlockCoord`: Calculates and returns the multidimensional grid coordinates (`blockCoord`) of a basic block based on its linear input block index. (Note: These coordinates are defined at block granularity, not element granularity.)
- `GemmCoord GetActualBlockShape`: Returns the actual boundary-aligned shape of a basic block (defined at element granularity) based on its input grid coordinates (`blockCoord`).

## Specific Swizzle Policies

| Component                       | Description|
| :----------------------------------------------------------- | :------: |
| [GemmIdentityBlockSwizzle](./GemmIdentityBlockSwizzle.md)    |   Basic swizzle policy for the GEMM operator |
