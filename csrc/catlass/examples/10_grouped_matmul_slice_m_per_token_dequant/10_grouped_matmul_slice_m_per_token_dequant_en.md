# CATLASS GMM_sliceM_perToken_Dequant

## Prototype Design

| Name          | Class     | Data Type | Dimensions         | Format | Description                                           |
| ------------- | --------- | --------- | ------------------ | ------ | ----------------------------------------------------- |
| matA          | inTensor  | int8      | [m, k]             | ND     | Left matrix                                           |
| matB          | inTensor  | int8      | [groupCount, n, k] | ND     | Right matrix, supports transposition                  |
| groupList     | inTensor  | int       | [groupCount]       | ND     | Group size in the m-axis direction, accumulation list |
| scale         | inTensor  | bf16      | [groupCount, n]    | ND     | perChannel quantization scale                         |
| perTokenScale | inTensor  | bf16      | [m]                | ND     | perToken quantization scale                           |
| matD          | outTensor | bf16      | [m, n]             | ND     | Output matrix                                         |

## Sample Implementation

The CATLASS GMM_sliceM_perToken_Dequant sample operator is implemented based on the CATLASS Gemm API to support the GMM operator of the Ascend Atlas A2 hardware. The operator structure consists of the following parts:

- **Example assembly**, [grouped_matmul_slice_m_per_token_dequant.cpp](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp);
- **Kernel implementation**, [grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp);
- **Block components**, including:
  - General mmad component [block_mmad_preload_async_with_callback.hpp](../../include/catlass/gemm/block/block_mmad_preload_async_with_callback.hpp);
  - Customized epilogue component [block_epilogue_per_token_dequant.hpp](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp);
- **Tile components**. In addition to the basic Tile_copy and tile_mmad components, pay attention to the following components:
  - `BlockMmad`[CopyGmToL1GMMPTD](../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)
  - [TileRowBroadcastMul](../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp#L32) in epilogue
  - [TileBroadcastOneBlk](../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp#L23) in epilogue
  - [TileOneBlkColumnBroadcastMul](../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp#L88) in epilogue

## Example Assembly

### Constructing the Input

- Calculate the [input data volume (len)] (../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L82) of each input.
- Calculate the [input data size (size)](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L88) of each input.
- [Generate the initial value of each input on the host](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L94). Use [GenerateGroupList()](../common/golden/fill_data.hpp#L70) to randomly generate `groupList` (prefix sum of the M axis).
- [Construct the input on the device](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L105).

### Assembling BlockMmad

- [Define the layout of each input](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L131). For details about the layout, see [here](../../docs/en/2_Design/02_tla/01_layout.md).
- [Set DispatchPolicy](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L152) to [MmadAtlasA2PreloadAsyncWithCallback](../../docs/en/2_Design/01_kernel_design/03_dispatch_policies.md), that is, select the BlockMmad component.
- Set [L1TileShape and L0TileShape](../../docs/en/2_Design/01_kernel_design/03_dispatch_policies.md) to split the basic block into cores and L1/L0 blocks. Note that there are structural constraints between the settings of `TileShape` and `DispatchPolicy`. For example, when L1::M=128 and L1::K=128, `l0AStages` cannot exceed 4 to avoid exceeding the L0A capacity size, which would otherwise be intercepted by static verification on the BlockMmad side during sample compilation. For details about `TileShape` computation, see [TileShape constraints](../../docs/en/1_Practice/11_matmul_optimization.md).
- [Set blockMmad input and output types](../../docs/en/2_Design/01_kernel_design/03_dispatch_policies.md).
- To specifically use `CopyGmToL1GMMPTD` to perform the `CopyGmToL1A` action, a new `TileCopyMmad` is [redefined](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L49) and [assembled](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L161). If `CopyGmToL1GMMPTD` is not enabled, this step can be skipped, and BlockMmad will use the default `TileCopyMmad`.
- [Assemble BlockMMAD](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L162) using the above template input parameters.

### Assembling blockEpilogue

- [Set EpilogueDispatchPolicy](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L166) to `EpilogueAtlasA2PerTokenDequant`, which selects the block_epilogue component.
- Define the [types of the input parameters related to epilogue](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L167): ScaleType, PerTokenScaleType, and DType.
- Define the [type for each computation step](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L171) in epilogue: RowBroadcastMulType, BroadcastOneBlkType, and OneBlkColumnBroadcastMulType.
- Define the [tile size for a single computation](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L175) in epilogue: <m0,n0>.
- Define the Tile used for epilogue computation, [TileRowBroadcastMul](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L176): For a single tile block <m0,n0> and scale segment <1,n0>, the scale segment <1,n0> is broadcast to <m0,n0> first, and then elementwise multiplication is performed.
- Define the Tile used for epilogue computation, [TileBroadcastOneBlk](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L177): The perTokenScale segment <m0,1> is broadcast to <m0, blk>.
- Define the Tile used for epilogue computation, [TileOneBlkColumnBroadcastMul](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L179): For a single tile block <m0,n0> and the broadcast segment <m0,blk> of perTokenScale, the perTokenScale segment <m0,blk> is broadcast to <m0,n0> first, and then elementwise multiplication is performed. Used together with [TileBroadcastOneBlk](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L177).
- Define the Tile used for epilogue copy, [TileCopy(../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L181). [Specialization position](../../include/catlass/epilogue/tile/tile_copy.hpp#L71).
- Define the swizzle policy [EpilogueHorizontalTileSwizzle](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L182) for tiling <m0,n0> when computing basic blocks in epilogue. [Source code location](../../include/catlass/epilogue/tile/tile_swizzle.hpp#L55). Currently, `EpilogueIdentityTileSwizzle` and `EpilogueHorizontalTileSwizzle` are provided.
- [Assemble BlockEpilogue](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L184) using the above template input parameters.

### Assembling and Executing the Kernel

- Define the [swizzle policy of the basic block](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L188) in the kernel. For details, see [the swizzle explanation](../../docs/en/2_Design/01_kernel_design/02_swizzle.md).
- [Assemble the kernel](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L192) GroupedMatmulSliceMPerTokenDequantMultiStageWorkspace, which needs to reference the corresponding [kernel file](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L29).
- [Assemble](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L195) the kernel into the adapter and [instantiate](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L201) it.
- [Construct the input arguments](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L197).
- [Verify the input arguments](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L202). Currently, this function is not implemented in the kernel and returns `true` directly.
- [Calculate the workspace size required by the operator](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L203), which is implemented in the kernel.
- [Allocate the workspace](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L206).
- [Initialize the adapter operator](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L208). Note that for operators involving inter-core synchronization, you need to [initialize fftsAddr](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L138) and pass it when executing the MatmulAdapter.

```cpp
// Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    ...
    matmulOp(stream, aicCoreNum, fftsAddr);
```

- [Execute the adapter operator](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L209).
- [Stream synchronization](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L210)

### Accuracy Verification and Space Release

- [Copy the operator output back to the host](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L213)
- [Compute the golden benchmark](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L216)
- [Verify precision](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L221)
- [Release the input, output, and workspace](../10_grouped_matmul_slice_m_per_token_dequant/grouped_matmul_slice_m_per_token_dequant.cpp#L228)

## Kernel Implementation

### Main structures and functions of the kernel

- [struct Params](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L61): parameters required for execution
- [struct Arguments](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L104): input parameters on the host side
- [static Params ToUnderlyingArguments](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L129): parses the host-side input Arguments into Params, which is called by the adapter during operator initialization
- [static size_t GetWorkspaceSize](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L121): calculates the workspace required for operator execution based on Arguments
- [void operator()`<AscendC::AIC>`](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L167): AIC execution code
- [void operator()`<AscendC::AIV>`](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L277): AIV execution code
- [struct AicWaitFunc](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L350): encapsulates the inter-core synchronization of the AIC waiting for the MTE3 transfer completion of the AIV. The MmadAtlasA2PreloadAsyncWithCallback solution is used on the AIC. Therefore, the callback needs to be passed into the blockMmad side, and the blockMmad determines when to invoke the inter-core synchronization.
- [struct AicSetFunc](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L367): encapsulates the inter-core synchronization of the AIV waiting for the completion of FIXPIPE transfer by the AIC.

### AIC Process

- Initialize BlockScheduler and BlockMmad. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L170)
- Initialize GlobalTensor: gmA/groupList/gmC. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L173)
- Obtain the current AIC sequence number (coreIdx) and the total number of AICs (coreNum). [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L179)
- Initialize the computation parameters: address offset of matrix A in the GM (gmGroupOffsetA), address offset of matrix B in the GM (gmGroupOffsetB), output data layout in the workspace (layoutC), stage ID and stageUsed corresponding to the WORKSPACE_STAGES operation, and start AIC sequence number of the current group (startCoreIdx). [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L181)
- Group loop. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L191)
  - Calculate M of the current group. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L192)
  - Determine whether to enable the L2 cache bypass for matrix A. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L204)
  - Calculate the start index of the current core in the tile of the current group. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L209)
  - Tile loop of the current core in the current group. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L211)
    - Calculate the input parameters required by blockMmad. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L213)
    - Call different blockMmads based on whether the asynchronous solution is used. ([code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L232)). In the current example, the asynchronous solution is used. The callback needs to be passed into blockMmad to schedule the invocation.
    - Update stageId. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L252)
  - Update gmGroupOffsetA, gmGroupOffsetB, and startCoreIdx. [Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L255)
- The asynchronous solution blockMmad completes the remaining computation.[Code](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant_multistage_workspace.hpp#L261)

The AIV execution flow is identical to that of the AIC, except that the call to `blockMmad()` is replaced by `blockEpilogue()`, and inter-core synchronization is performed within the kernel code via the callback component.

### Basic Block Splitting by Core

- Basic blocks are split in each group (with the size of `[currentM, N]`) according to the size of `[L1TileShape::M, L1TileShape::N]`.
- Cores are assigned continuously across groups to achieve load balancing among different AICs.
- Related variables:
  - `groupIdx`: ID of the current group
  - `coreLoops`: number of blocks in the current group
  - `startCoreIdx`: ID of the AIC of the start block (in the current group)
  - `startLoopIdx`: ID of the start block of the current core (in the current group)
  - `loopIdx`: ID of the block to be processed by the current core (in the current group)
- Example:

<img src="https://raw.gitcode.com/user-images/assets/7801479/6029234c-39e4-4853-99de-1d4263f4e91f/Block_Partitioning_Scheme.png" width="100%">

### Workspace Scheme

Under the current configuration of `workspaceStages=2`, double-buffering of tiles is used, where each AIC is allocated a workspace equal to the size of two tiles. For inputs of any shape, the requested workspace size remains fixed.

```cpp
static size_t GetWorkspaceSize(const Arguments &args)
{
    size_t lenWorkspace = static_cast<size_t>(L1TileShape::M) * L1TileShape::N *
        args.aicCoreNum * WORKSPACE_STAGES;
    size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);
    return sizeWorkspace;
}
```

(Another scheme available in the library allocates a full workspace of `problemShape.m * problemShape.n`, which can save workspace memory in small-shape scenarios. See [Reference](../../include/catlass/gemm/kernel/grouped_matmul_slice_m_per_token_dequant.hpp).)

<img src="https://raw.gitcode.com/user-images/assets/7801479/8b950918-80c1-4e8d-a0de-63a6a4b6cf56/workspace.png" width="100%">

### Inter-AIC/AIV Synchronization Scheme

On the AIC, the Mmad operation for a single block can be split into three actions: L1Tile loading, L1Tile computation, and L0C storing. Due to the preloading and asynchronous nature of blockMmad, when `blockMmad()` is invoked inside `operator()<AscendC::AIC>`, it only completes the L1Tile loading and most of the L1Tile computation for the current block. The remaining L1Tile computation and L0C storing are executed during the subsequent invocation of `blockMmad()`. Therefore, the callback arguments for inter-AIC/AIV synchronization must be passed into `blockMmad()`, which determines when to trigger them.

The flowchart below illustrates a single AIC and its two allocated AIVs processing blocks sequentially on the GM:

<img src="https://raw.gitcode.com/user-images/assets/7801479/79aadfa0-3568-4cbc-908f-a90ec7b1cfa3/AIV_AIC同步.png" width="100%">
