# Block Mmad Pingpong
>
> [Code location](../../../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp)

## Description

A partial specialization implementation of [BlockMmad](./block_mmad.md#blockmmad) designed for block-level MMAD computation. It does not perform bias computation, operates non-asynchronously, and does not use the TLA implementation.

## Scheduling Policy

```
// Now ENABLE_UNIT_FLAG_ must be false when input element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};
```

When `ENABLE_UNIT_FLAG_` is set to `true`, concurrent read-out and write-in for `L0C` are enabled, maximizing pipeline parallelism. **Note: When the input data type is int8, this parameter must be set to `false`. Otherwise, computation exceptions may occur.**

## Example

### Block Assembly

See [basic_matmul ](../../../../../../../ examples/00_basic_matmul/basic_matmul.cpp).

```
constexpr bool enableUnitFlag = true;
using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<half, LayoutA>;
using BType = Gemm::GemmType<half, LayoutB>;
using CType = Gemm::GemmType<half, LayoutC>;
```

```
using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
```

### Block Instantiation

Executed inside the `void operator()<AscendC::AIC>` core function of the kernel code by referring to [basic_matmul](../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp).

```
Arch::Resource<ArchTag> resource;
BlockMmad blockMmad(resource);
```

### Block Execution

Executed inside the `void operator()<AscendC::AIC>` core function of the kernel code by referring to [basic_matmul](../../../../../../../include/catlass/gemm/kernel/basic_matmul.hpp).

```
blockMmad(gmA[gmOffsetA],       // GM start address of the tile block for matrix A
        params.layoutA,         // Storage layout of matrix A in GM
        gmB[gmOffsetB],         // GM start address of the tile block for matrix B
        params.layoutB,         // Storage layout of matrix B in GM
        gmC[gmOffsetC],         // GM start address of the tile block for matrix C
        params.layoutC,         // Storage layout of matrix C in GM
        actualBlockShape);      // Actual structural shape of the block
```

## Constraints

- The template parameter `BiasType_` is not used in practice and does not support bias computation.
