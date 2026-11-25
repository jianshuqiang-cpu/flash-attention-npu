# Gemm Identity Block Swizzle
>
> [Code location](../../../../../../../../include/catlass/gemm/block/block_swizzle.hpp)

## Description

The swizzle policy determines the allocation relationship and computation sequence of the basic task blocks across the AI Cores. For a deeper breakdown of the underlying mechanics, see [Swizzle Policies](../../../../../../2_Design/01_kernel_design/02_swizzle.md). The `GemmIdentityBlockSwizzle` policy partitions the destination matrix C into basic blocks along the $M$ and $N$ dimensions. The exact scheduling of these blocks is governed by `SwizzleOffset` and `SwizzleDirection`. Once the sequence is calculated, the blocks are mapped to physical AI Cores sequentially according to their physical core IDs.

As illustrated in the example below, assuming a hardware environment with 20 AI Cores, the basic blocks are assigned to the cores one by one along the path of the arrows. The traversal order is determined by configuring `SwizzleOffset = 1` and `SwizzleDirection = 0`.

<img src="https://raw.gitcode.com/user-images/assets/7801479/bc97a077-d2e2-4abd-8f31-55316b1e0906/image.png" width="60%">

## Common Methods

| Return Type  | Function Name| Input Parameters|Functions|
| :------- | :------: |:------: |:------: |
| -  |  GemmIdentityBlockSwizzle  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_  |Constructor|
| -  |  GemmIdentityBlockSwizzle  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_  |Constructor|
| void |  Update  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_  |Updates `problemShape`, `tileMN` and `loopsMN`|
| void |  Update  |GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_  |Updates `problemShape`, `tileMN` and `loopsMN`|
| uint32_t |  GetCoreLoops  |-  |Returns the total number of basic blocks (the product of the block counts in the $M$ and $N$ dimensions)|
| uint32_t |  GetBatchIdx  |uint32_t taskIdx  |Calculates the batch ID to which the basic block corresponding to the input taskIdx belongs|
| GemmCoord |  GetBlockCoord  |uint32_t taskIdx  |Calculates the coordinates (`blockCoord`) across each dimension for the basic block corresponding to the input taskIdx.|
| GemmCoord |  GetActualBlockShape  |GemmCoord blockCoord  |Returns the actual shape of the basic block based on its coordinates (`blockCoord`)|

## Example

### Block Assembly

See [basic_matmul](../../../../../../../../examples/00_basic_matmul/basic_matmul.cpp).

```
// Swizzle offset is 3 and direction is 0.
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// kernel level
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
```

### Using Matmul Kernels

In the `void operator()<AscendC::AIC>` function of the kernel code (see [basic_matmul](../../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)):

Instantiate BlockScheduler:

```
BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
```

Obtain the total loop count for the blocks:

```
uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
```

Obtain the actual coordinates and shape of the current `block`:

```
for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        // Compute block location
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
        ...
}
```

### Using GroupMatmul Kernels

In the `void operator()<AscendC::AIC>` function of the kernel code (see [grouped_matmul_slice_m_per_token_dequant_multistage_workspace](../../../../../../../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp)):

Instantiate BlockScheduler:

```
BlockScheduler blockScheduler;
```

During group traversal, update the `blockScheduler` based on the shape of each group, and obtain the total number of basic blocks for a single group.

```
for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
    ...
    blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
    uint32_t coreLoops = blockScheduler.GetCoreLoops();
    ...
}
```

During the traversal of basic blocks within a single group, obtain the block coordinates and actual shape of the current block.

```
for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
        GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);
        ...
}
```
