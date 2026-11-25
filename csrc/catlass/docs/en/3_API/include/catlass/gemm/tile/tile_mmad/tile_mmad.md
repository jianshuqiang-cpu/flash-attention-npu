# TileMmad

> [Code Location](../../../../../../../../include/catlass/gemm/tile/tile_mmad.hpp)

[TOC]

## Function

`TileMmad` utilizes basic API [AscendC::Mmad] (https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html) to perform the matrix multiply-accumulate operation `C += A * B`. The operands reside in the following cache locations: A in L0A, B in L0B, and C in L0C, with fixed data layouts of zZ, nZ, and zN respectively. This is a non-TLA implementation.

Two calling modes are supported:
- Without Bias: standard matrix multiply-accumulate
- With Bias: Loads the bias from the Bias Tensor (BT) into L0C before executing matrix multiply-accumulate.

Unlike the TLA-style version [TileMmadTla](./tile_mmad_tla.md), this template operates directly on `AscendC::LocalTensor` and does not use the `tla::Tensor` wrapper.

### Architecture Differences

| Architecture| `kDirectionAlign` | `disableGemv` | Description|
| :------ | :------ | :------ | :------ |
| Atlas A2 (2201)| Enabled for `float` + `ColumnMajor`/`nZ` L1A layout| — | K-direction alignment optimization|
| Ascend 950 (3510) | — | false when L1A is `VectorLayout`; otherwise true| GEMV mode control|

## Template Prototype

```cpp
template <
    class ArchTag_,            // Architecture tag
    class AType_,              // A matrix GmType
    class BType_,              // B matrix GmType
    class BiasType_            // Bias GmType
>
struct TileMmad;
```

## APIs

### Without Bias

```cpp
void operator()(
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,   // L0C accumulation result
    AscendC::LocalTensor<ElementA> const &l0ATensor,             // L0A left matrix
    AscendC::LocalTensor<ElementB> const &l0BTensor,             // L0B right matrix
    uint32_t m,             // M dimension (after alignment)
    uint32_t n,             // N dimension (after alignment)
    uint32_t k,             // K dimension (after alignment)
    bool initC = true,      // true= overwriting, false = atomic addition
    uint8_t unitFlag = 0    // L0C→Global Memory parallel data movement flag
);
```

### With Bias

```cpp
void operator()(
    AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
    AscendC::LocalTensor<ElementA> const &l0ATensor,
    AscendC::LocalTensor<ElementB> const &l0BTensor,
    AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,  // BT Bias data
    uint32_t m, uint32_t n, uint32_t k,
    bool initC = true, // Forcibly set to false when bias is used (internal overwriting)
    uint8_t unitFlag = 0
);
```

### Tail Pipeline Barrier

A `PipeBarrier<PIPE_M>()` is automatically inserted when `(m/16) * (n/16) < 10`, in order to prevent pipeline conflicts.

## Examples

### Without Bias

```cpp
#include "catlass/gemm/tile/tile_mmad.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::zZ>;
using BType = Gemm::GemmType<half, layout::nZ>;
using BiasType = void;

AscendC::LocalTensor<half> l0ATensor;
AscendC::LocalTensor<half> l0BTensor;
AscendC::LocalTensor<float> l0CTensor;

Tile::TileMmad<Arch::AtlasA2, AType, BType, BiasType> mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32);
```

### With Bias

```cpp
using AType   = Gemm::GemmType<half, layout::zZ>;
using BType   = Gemm::GemmType<half, layout::nZ>;
using BiasType = Gemm::GemmType<float, layout::VectorLayout>;

AscendC::LocalTensor<float> l0BiasTensor;

Tile::TileMmad<Arch::AtlasA2, AType, BType, BiasType> mmadOp;
mmadOp(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, 64, 64, 32);
```

### unitFlag parallel data movement

```cpp
bool initC   = true;
uint8_t unitFlag = 1; // Enable parallel data movement from L0C to Global Memory.

// The first mmad operation initializes the accumulator C.
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32, initC, unitFlag);

// Subsequent mmad operations: atomic accumulation + continued parallel movement
initC = false;
mmadOp(l0CTensor, l0ATensor, l0BTensor, 64, 64, 32, initC, unitFlag);
```
