# Block Epilogue Visitor Partial Specialization
>
> Code location: `include/catlass/epilogue/block/block_epilogue_visitor.hpp`

## Description

This is the `BlockEpilogue` partial specialization implementation used by the EVG.

It partitions the macro Block result into smaller Tiles and drives the EVG to process each Tile in a strict `LOAD -> COMPUTE -> STORE` sequence. Additionally, it coordinates a double-buffered pipeline by alternating between two sets of callbacks.

## Template Form

The `BlockEpilogue` configuration utilized by the EVG takes the following forms:

```cpp
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueVisitor<false>,
    ArchTag,
    Int<computeLength>,
    EVG,
    ElementC
>;
```

or:

```cpp
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueVisitor<true>,
    ArchTag,
    Int<computeLength>,
    EVG,
    ElementC
>;
```

The corresponding partial specialization template is defined as:

```cpp
template <
    bool USE_UB_WORKSPACE_,
    class ArchTag_,
    class ComputeLength_,
    class EVG_,
    class ElementC_
>
class BlockEpilogue<
    EpilogueVisitor<USE_UB_WORKSPACE_>,
    ArchTag_,
    ComputeLength_,
    EVG_,
    ElementC_
>;
```

## Key Responsibilities

- Receives `EVG::Params`.
- Partitions the Block calculation results into Tiles.
- Allocates two independent sets of callbacks for the EVG.
- Drives Tile execution following the `LOAD -> COMPUTE -> STORE` sequence.
- Implements Tile-level double buffering using hardware event synchronization.

## Key Parameters

- `EpilogueVisitor<false>`: Fetches the MMAD result from the GM workspace.
- `EpilogueVisitor<true>`: Fetches the MMAD result directly from the UB.
- `ComputeLength`: The number of elements processed per Tile operation; must satisfy alignment constraints.
- `EVG`: Entire epilogue graph.
- `ElementC`: Type of the MMAD output.

## Key Execution Points

- When `USE_UB_WORKSPACE == false`, the EVG reads the matrix $C$ data from the GM workspace.
- When `USE_UB_WORKSPACE == true`, the EVG directly consumes the MMAD results in the UB.
- The internal logic dynamically switches between multi-row full-column processing and row-by-row split-column processing based on the active Tile width configuration.
- The two callback sets cycle alternately to establish a double-buffering scheme at the Tile granularity.

## References

- [block_epilogue](./block_epilogue.md)
- [fusion/README](../fusion/README.md)
- [evg_api](../../../../evg_api.md)
