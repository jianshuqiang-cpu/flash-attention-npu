# Gemm/Block Class Template Overview

## API List

### blockMmad List

| Component                      | Description|
| :----------------------------------------------------------- | :------: |
| [block_mmad](./block_mmad.md#blockmmad)     |   Basic template, including BlockMmad. |
| [block_mmad_pingpong](./block_mmad_pingpong.md)     |   Partial specialization of BlockMmad implementing ping-pong matrix multiplication. |

### Swizzle List

| Component                       | Description|
| :----------------------------------------------------------- | :------: |
| [block_swizzle](./block_swizzle/block_swizzle.md)    |   Basic swizzle methods |
| [GemmIdentityBlockSwizzle](./block_swizzle/GemmIdentityBlockSwizzle.md)    |   Basic swizzle policy for the GEMM operator |

## API Breakdown

### blockMmad
>
>The blockMmad structure encapsulates the MMAD computation at the Block layer, mapping directly to execution on a single AI Core of the Ascend NPU. Through template parameters, it receives configuration details defining the matrix shapes, tensor layouts (such as row-major or column-major), and data types (DType).

The namespace is `Catlass::Gemm::Block`. Core members:

| Type        | Name|Function|
| :----------------------------------------------------------- | :------: |------: |
| Constructor   |   BlockMmad()  |Initializes buffers, registers event IDs, and inserts `setFlag` primitives for pipeline synchronization.|
| Destructor   |   ~BlockMmad()  |Inserts `waitFlag` primitives for pipeline synchronization.|
| Function   |   void operator()  |Executes the matrix multiplication for a Block task.|
