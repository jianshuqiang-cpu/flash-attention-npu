# Block Epilogue Per Token Dequant
>
> [Code location](../../../../../../../include/catlass/epilogue/block/block_epilogue_per_token_dequant.hpp)

## Description

- This is a [BlockEpilogue](./block_epilogue.md) partial specialization that performs per-token and per-channel dequantization on block data using `perTokenScale` and `perChannelScale`.

- Formula: $blockD_{ij} = blockC_{ij} * perChannelScale_j * perTokenScale_i$
- Currently supported data types for `blockC`, `perChannelScale`, `perTokenScale`, and `blockD`

<table><thead>
  <tr>
    <th rowspan="2">blockC</th>
    <th rowspan="2">perChannelScale</th>
    <th rowspan="2">perTokenScale</th>
    <th rowspan="2">blockD</th>
  </tr></thead>
  <tbody>
  <tr>
    <td>int32</td>
    <td>half</td>
    <td>half</td>
    <td>half</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>bfloat16_t</td>
    <td>bfloat16_t</td>
    <td>bfloat16_t</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>float</td>
    <td>float</td>
    <td>half</td>
  </tr>
  <tr>
    <td>int32</td>
    <td>float</td>
    <td>float</td>
    <td>bfloat16_t</td>
  </tr>
</tbody>
</table>

## Dispatch Policy

```
// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};
```

## Example

### Block Assembly

See [Sample 12_quant_matmul](../../../../../../../examples/12_quant_matmul/quant_matmul.cpp).

```
constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
using ScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
using PerTokenScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

using EpilogueTileShape = MatrixShape<32, 256>;
using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
using TileBroadcastOneBlk =
    Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
using TileOneBlkColumnBroadcastMul =
    Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
```

```
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy,        // Epilogue dispatch policy configuration
    CType,                         // Block tensor type before dequantization
    ScaleType,                     // Tensor type for perChannelScale
    PerTokenScaleType,             // Tensor type for perTokenScale
    DType,                         // Block tensor type after dequantization
    TileRowBroadcastMul,           // Tile component: broadcasts (1, n) scale to (m, n) and multiplies with block
    TileBroadcastOneBlk,           // Tile component: broadcasts (m, 1) perTokenScale to (m, 32B)
    TileOneBlkColumnBroadcastMul,  // Tile component: broadcasts (m, 32B) perTokenScale to (m, n) and multiplies with block
    TileCopy,                      // TileCopy execution component
    TileScheduler                  // Tile partitioning scheduler
>;
```

### Block Instantiation

In the `void operator()<AscendC::AIV>` function of the kernel code (see [quant_matmul_multistage_workspace](../../../../../../../include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp)):

```
BlockEpilogue blockEpilogue(resource);
```

### Updating Parameters

In the `void operator()<AscendC::AIV>` function of the kernel code (see [quant_matmul_multistage_workspace](../../../../../../../include/catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp)):

```
EpilogueParams epilogueParams{
    params.ptrScale,          // GM address of perChannelScale
    layoutScale,              // Tensor layout of perChannelScale
    params.ptrPerTokenScale,  // GM address of perTokenScale
    layoutPerTokenScale,      // Tensor layout of perTokenScale
    params.ptrD,              // GM address of the output matrix
    layoutD                   // Tensor layout of the output matrix
};

blockEpilogue.UpdateParams(epilogueParams);
```

### Block Execution

In the `void operator()<AscendC::AIC>` function of the kernel code (see [basic_matmul](../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp)):

```
blockEpilogue(
    blockShapeMNK,          // Shape of the block
    blockCoordMNK,          // Block coordinates in the destination matrix (block granularity)
    actualBlockShapeMNK,    // Runtime shape of the target block being processed
    gmBlockC,               // GM start address of the target block
    layoutBlockC            // Tensor layout of the target block
);
```

## Constraints

- Only `RowMajor` configurations are supported for blockC and blockD. `VectorLayout` configurations are supported for `perChannelScale` and `perTokenScale`.
