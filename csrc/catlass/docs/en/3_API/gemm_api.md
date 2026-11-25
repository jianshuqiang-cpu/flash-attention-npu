# CATLASS Gemm API

CATLASS provides a unified programming model for Matrix Multiply-Accumulate (MMAD) operations executed across different hierarchy levels on the NPU. The CATLASS GEMM API maps to the following layered abstraction structure, ordered from highest to lowest.
![image](../../assets/images/api_level_en.png)

# CATLASS Gemm Model

Based on the layered abstraction structure shown above, CATLASS implements the classic three-layer nested loop matrix multiplication algorithm.

The following pseudocode defines the execution model of a Matmul kernel targeting intra-core synchronous matrix multiplication hardware instructions (such as `mmad`). This is pseudocode and is only used to illustrate which parts of the layers correspond to the inner or outer loops of the matrix multiplication.

```c++
// Catlass::Gemm::Kernel::BasicMatmul: BlockTileM and BlockTileN loops
// Parallel on AI Cores
for (int block_m = 0; block_m < MatmulM; block_m += BlockTileM) {
  for (int block_n = 0; block_n < MatmulN; block_n += BlockTileN) {

    // Catlass::Gemm::Block::BlockMmad: main loop iterated over k-tiles
    // No loop unrolling in this phase
    for (int k_tile = 0; k_tile < MatmulK; k_tile++) {

      // Inner loop at the tile level: (m,k) x (k,n) => (m,n)
      // TileMmad uses the hardware instruction AscendC::Mmad.
      for (int tile_mma_m = 0; tile_mma_m < m; tile_mma_m++) {
        for (int tile_mma_n = 0; tile_mma_n < n; tile_mma_n++) {
          for (int tile_mma_k = 0; tile_mma_k < k; tile_mma_k++) {
            mmad.call(c, a, b);
          } // tile_mma_k
        } // tile_mma_n
      } // tile_mma_m
    } // k_tile mainloop
  } // block_n
} // block_m
```

The first two nested `for` loops correspond to parallelism across multiple AI Cores. The code does not explicitly express them as two `for` loops. Instead, the data blocks processed by different cores are distinguished by `BlockIdx`.

Within the two nested `for` loops, global memory (GM) is tiled, and the tiles are moved to more local memory (such as L1 Buffer or L0 Buffer) to execute MMAD computations. The iterations of these tile copies and tile MMAD computations are typically completely static and fully unrolled.

# CATLASS Gemm Components

CATLASS uses the following components to express the loop nesting described above. These components are specialized based on data types, data layouts, and mathematical instructions.

| API Level            | API Class and/or Function Name                                                        |
| -------------------- | ------------------------------------------------------------------------------------- |
| Device               | `Catlass::Gemm::Device::DeviceGemm`                                                   |
| Kernel               | `Catlass::Gemm::Kernel::BasicMatmul`                                                  |
| Block                | `Catlass::Gemm::Block::BlockMmad` <br> `Catlass::Epilogue::Block::BlockEpilogue` <br> |
| Tile (MMAD and Copy) | `TileMmad` and `TileCopy` <br>                                                        |
| Basic                | `AscendC::Mmad` and `AscendC::DataCopy`                                               |

In CATLASS, kernels are assembled by first combining the Block main loop and Block epilogue at the Kernel layer, and then wrapping them with a host-side adapter.

When assembling a kernel using these components, instantiation must follow this sequence:

1. Configure the required Block main loop and Block epilogue.
2. Combine these blocks to construct the Kernel.
3. Wrap the Kernel using a Device layer adapter.

This order is demonstrated in the CATLASS example [examples/00_basic_matmul](../../../examples/00_basic_matmul), as shown in the following code excerpt:

```c++
// Step 1: Create the required specialized block-level MMAD
// Parameters.
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy,
    L1TileShape,
    L0TileShape,
    AType,
    BType,
    CType>;

// Step 2: (Optional) Specify the block-level epilogue type
using BlockEpilogue = void;

// Step 3: Specify the block scheduling mode for data movement
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<>;


// Step 4: Combine MMAD and epilogue at the kernel layer
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// Step 5: Wrap the kernel in the device adapter for host-side invocation
using Matmul = Catlass::Gemm::Device::DeviceGemm<MatmulKernel>;
```

## Block API

The Block API encompasses matrix multiply-accumulate operations and the epilogue. It is responsible for implementing the `k_tile` loop specified in the three-layer nested loop execution model.

In the SPMD programming model of the Ascend NPU, a Block represents a process, which corresponds to a logical core. It abstracts the following hardware features:

- Asynchronous memory copy (for example, from GM to L1 Buffer)
- MMAD instructions on tile-granularity data residing in the L0 Buffer
- Synchronization operations: Managing coordination across multiple cores as well as between different hardware pipelines within a single core to ensure asynchronous data dependencies are met.

A Block utilizes the `TileMma` and `TileCopy` APIs (detailed below) to perform tile-granularity data transfers and MMAD computations.

Different hardware pipelines within a block (such as MTE1, MTE2, or FixPipe) provide specialized capabilities. These pipelines must share data and synchronize access to shared data. For example, after MTE2 copies data from GM to the L1 Buffer, it must notify MTE1 that the input data is ready. The `Kernel::` layer handles the orchestration by invoking the `Block::` layer interfaces, while the `Block::` layer itself manages the independent computation of individual matrix C blocks.

### Block Mmad

`Catlass::Gemm::Block::BlockMmad` is the primary interface for the `MMAD` main loop at the Block layer.

The `BlockMmad` class is defined in the header file
[include/catlass/gemm/block/block_mmad.hpp](../../../include/catlass/gemm/block/block_mmad.hpp).

```c++
namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {};

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

```

- `DispatchPolicy`: A critical configuration parameter for the Block layer, detailed in the subsequent section.
- `L1TileShape` and `L0TileShape`: Define the base tile dimensions utilized within the L1 Buffer and L0 Buffer, respectively. These are covered in detail later.
- `AType`, `BType`, `CType`, and `BiasType`: Specialized instances of `GemmType` that encapsulate the data types and data layouts of the A, B, and C matrices, along with the bias vector in GM.
- `TileCopy`: An instance of `Tile::TileCopy` that encapsulates tile-granularity data transfers between different memory hierarchy levels, for example, GM to L1 Buffer, and L1 Buffer to L0 Buffer.
- `TileMmad`: An instance of `Tile::TileMmad` that executes the matrix multiply-accumulate operations at the base tile granularity within the L0 Buffer.

### Block Dispatch Policies

The implementation of `BlockMmad` is not universal. Instead, it must be specialized for each algorithm and NPU architecture. Users dispatch to a specific `BlockMmad` specialization by selecting the template parameters that match that specialization.
CATLASS adopts a tag-based dispatch policy design to specialize the Block layer MMAD implementations and provide tuning capabilities.

The following is an example of a dispatch policy targeting the Atlas A2 architecture, which utilizes a ping-pong buffer mechanism within the L1 Buffer and enables `unitflag` optimization:

```c++
// 2-Buffer in L1 Buffer ,
// unitflag enable
struct MmadAtlasA2Pingpong {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = true;
};
```

The `STAGES` parameter allows users to easily adjust the number of buffers in multi-buffer scenarios. The `ENABLE_UNIT_FLAG` parameter indicates whether to enable fine-grained parallelism between MMAD computation and the data transfers of L0C results back to GM.

Adopting this dispatch policy design provides the following advantages:

- Eliminates code duplication: The main loop can be reused across different kernels.
- Simplifies generic programming: The core structure name `BlockMmad` remains identical across all implementations.
- Provides a single extension point: It offers a clean, unified boundary for users to insert new, custom main loops specialized for their own dispatch policies.

### TileShape

`L1TileShape` and `L0TileShape` correspond to the base tile dimensions utilized within the L1 Buffer and L0 Buffer, respectively, and are defined by $(m, n, k)$.

### Epilogue

The epilogue implements element-wise operations involving the output matrix. Users can provide a custom epilogue or utilize one of the standard pre-configured options. These components are located in the [include/catlass/epilogue/block/](../../../include/catlass/epilogue/block/block_epilogue.hpp) directory and include classes such as `Catlass::Epilogue::Block::BlockEpilogue`. The epilogues provided by CATLASS are intentionally decoupled from the [include/catlass/gemm](../../../include/catlass/gemm) and the `Catlass::Gemm` namespace because they can be leveraged for non-GEMM compute operations.

## Kernel API

The Kernel layer encapsulates the collective execution logic of all Blocks scheduled on the NPU. Specifically, `BasicMatmul` at the Kernel layer provides the following functionality:

- Combines the individual logic of different Blocks and injects necessary synchronization primitives.
- Manages Block swizzling, which defines the mapping and relationship between different Blocks and their target data partitions in GM.
- Tiles and fragments the input data at Block granularity.

The Kernel layer API serves as the entry point for device-side execution and acts as the integration point for fusing sequential matrix multiplications, epilogues, or other custom operations.

The kernel API entry is defined with
`Catlass::Gemm::Kernel::BasicMatmul` in the header file
[include/catlass/gemm/kernel/basic_matmul.hpp](../../../include/catlass/gemm/kernel/basic_matmul.hpp).
`BasicMatmul` is a stateless device-side kernel. The matrix multiplication it implements consists of two parts:

- Block Mmad
- Block Epilogue

```cpp
namespace Catlass::Gemm::Kernel {
template <
  class BlockMmad_,
  class BlockEpilogue_,
  class BlockScheduler_
>
class BasicMatmul;
} // namespace Catlass::Gemm::Kernel
```

Note: "Stateless" indicates that the caller manages the state of the kernel, for example, through the Device API described below. The kernel itself only accepts input and output configuration parameters (`Params`).

Within this structure, Block MMAD represents the matrix multiply-accumulate operation performed on local data Blocks, while the Block Epilogue handles operations following the MMAD phase, such as the `beta × C` adjustment in the equation `$C := \beta \cdot C + \alpha \cdot A \cdot B$`.

## Device API

The Device layer serves as the entry point for host-side invocations, abstracting away the low-level complexities of launching device-side functions. Once the user configures the structural composition of the Kernel, it is passed into the Device layer template to execute the operator.

```cpp
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue = void;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<>;

// kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// device
using Matmul = Gemm::Device::DeviceGemm<MatmulKernel>;
Matmul matmulOp;
// 'args' encapsulates runtime parameters such as tensor dimensions
matmulOp(args, workspace, stream);
```

## Tile MMAD and Copy

Tile-level MMAD and copy operations represent structural combinations of the underlying mathematical and data transfer interfaces found in the Basic layer. The purpose of this layer is to construct composable NPU microkernels out of hardware-accelerated math operations and data movement operations, each specialized for specific data types and layouts. This layer provides a unified Interface that preserves identical computation or data transfer semantics across different hardware backends.

These operations are invoked in the innermost loops of the three-layer nested loop execution model using interfaces such as `Tile::TileMmad()`, `Tile::CopyGmToL1()`, or `Tile::CopyL0CToGm()`.

This layer is explicitly designated as the Tile level because it leverages the atomic primitives of the Basic layer to construct larger-granularity operations. As a reusable component, it mirrors the process of fitting individual tiles together to form a larger mosaic pattern.

## Basic API

The Basic layer encapsulates direct hardware instruction calls that accelerate MMAD and data copy operations. As the lowest foundational layer of CATLASS, it abstracts raw hardware capabilities to expose chip-level power while ensuring structural completeness and compatibility. Note that ISASI-level APIs in this layer do not guarantee cross-hardware-version compatibility.
