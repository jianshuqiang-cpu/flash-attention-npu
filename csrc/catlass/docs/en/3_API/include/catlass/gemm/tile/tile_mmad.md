# Basic Tile MMAD Template
>
> [Code location](../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## TileMmad

### Description

Performs matrix multiply-accumulate ($C += A \times B$) operations using the [AscendC::mmad basic API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html). Matrices A, B, and C correspond to data residing in L0A, L0B, and L0C, respectively. The data layout formats for the A, B, and C matrices are zZ, nZ, and zN, respectively. This is a non-TLA implementation.

### Prototype

- Structure template

```
template <
    class ArchTag_,         // Architecture tag
    class AType_,           // GEMM type for matrix A operand
    class BType_,           // GEMM type for matrix B operand
    class BiasType_         // GEMM type for bias operand
>
struct TileMmad
```

- Invocation without bias input

```
void operator() (
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,  // Destination matrix in L0C
    AscendC::LocalTensor<ElementA> const &l0ATensor,            // Left matrix in L0A
    AscendC::LocalTensor<ElementB> const &l0BTensor,            // Right matrix in L0B
    uint32_t m,             // M alignment of the actual tile block shape
    uint32_t n,             // N alignment of the actual tile block shape
    uint32_t k,             // K alignment of the actual tile block shape
    bool initC = true,      // Whether to initialize L0C. True overwrites data directly; False performs atomic accumulation.
    uint8_t unitFlag = 0    // Whether to enable unitFlag. If True, MMAD computation and L0C-to-GM data transfer run in parallel.
    )
```

- Invocation with bias input

```
void operator() (
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,  // Destination matrix in L0C
    AscendC::LocalTensor<ElementA> const &l0ATensor,            // Left matrix in L0A
    AscendC::LocalTensor<ElementB> const &l0BTensor,            // Right matrix in L0B
    AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,  // Bias in BT
    uint32_t m,             // M alignment of the actual tile block shape
    uint32_t n,             // N alignment of the actual tile block shape
    uint32_t k,             // K alignment of the actual tile block shape
    bool initC = true,      // Whether to initialize L0C. True overwrites data directly; False performs atomic accumulation.
    uint8_t unitFlag = 0    // Whether to enable unitFlag. If True, MMAD computation and L0C-to-GM data transfer run in parallel.
    )
```

### Example

For details, see the usage method in [block_mmad_pingpong](../../../../../../../include/catlass/gemm/block/block_mmad_pingpong.hpp).

## TileMmadTla

### Description

Performs matrix multiply-accumulate ($C += A \times B$) operations using the [AscendC::mmad basic API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html). Matrices A, B, and C correspond to data residing in L0A, L0B, and L0C, respectively. The data layout formats for the A, B, and C matrices are zZ, nZ, and zN, respectively. This is a TLA implementation and does not support bias.

### Prototype

- Structure template

```
template <
    class ArchTag_,         // Architecture tag
    class ElementA,         // Element type for matrix A operand
    class LayoutTagL1A      // Layout tag for matrix A operand
>
struct TileMmadTla
```

- Invocation (bias not supported)

```
void operator() (
    TensorC const &l0CTensor,   // Destination matrix tensor in L0C
    TensorA const &l0ATensor,   // Left matrix tensor in L0A
    TensorB const &l0BTensor,   // Right matrix tensor in L0B
    uint32_t m,                 // M alignment of the actual tile block shape
    uint32_t n,                 // N alignment of the actual tile block shape
    uint32_t k,                 // K alignment of the actual tile block shape
    bool initC = true,          // Whether to initialize L0C. True overwrites data directly; False performs atomic accumulation.
    uint8_t unitFlag = 0        // Whether to enable unitFlag. If True, MMAD computation and L0C-to-GM data transfer run in parallel.
    )
```

### Example

For details, see the usage method in [block_mmad_pingpong_tla](../../../../../../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp).
