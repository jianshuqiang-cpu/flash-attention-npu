# Block MMAD Basic Template
>
> [Code location](../../../../../../../include/catlass/gemm/block/block_mmad.hpp)

[TOC]

## BlockMmad

### Description

Block-level MMAD computation, non-TLA implementation. The actual computation is carried by the partial specialization templates, and the partial specialization branch is hit through template parameters such as [DispatchPolicy](../../../../../../../include/catlass/gemm/dispatch_policy.hpp).

### Template Description

```
template <
    class DispatchPolicy,   // Dispatch policy used
    class L1TileShape,      // L1 basic block
    class L0TileShape,      // L0 basic block
    class AType,            // Encapsulates the data type and layout information of matrix A
    class BType,            // Encapsulates the data type and layout information of matrix B
    class CType,            // Encapsulates the data type and layout information of matrix C
    class BiasType = void,  // Encapsulates the data type and layout information of bias
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>, // Tile-level transfer
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>          // Tile-level MMAD computation
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};
```

Note that among the template parameters, `BiasType` defaults to `void`, `TileCopy` defaults to [Gemm::Tile::TileCopy](../tile/tile_copy/README.md), and `TileMmad` defaults to [Gemm::Tile::TileMmad](../tile/tile_mmad/README.md).

## BlockMmadTla

(To be completed)

## BlockGemm

(To be completed)
