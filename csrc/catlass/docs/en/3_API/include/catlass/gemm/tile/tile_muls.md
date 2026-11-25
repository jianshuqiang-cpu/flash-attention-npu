# TileMuls

> [Code Location](../../../../../../../include/catlass/gemm/tile/tile_muls.hpp)

[TOC]

## Function

`TileMuls` is a template that implements scalar multiplication for the Vector engine using `AscendC::Muls`. It multiplies every element of a tensor residing in Unified Buffer (UB) by a given `scalar` and writes the result back to UB.

Application scenario: Scaling the accumulated results in the Epilogue phase, after data has been moved from L0C to UB

The template combines `AscendC::SetVectorMask` and `AscendC::Muls` with the COUNTER mask mode to accurately control the vectorized computation length, ensuring that only valid elements are processed and avoiding out-of-bound memory access.

## Template Prototype

```cpp
template <
    class ArchTag_,              // Architecture tag
    class ComputeType_,          // Computation type: Gemm::GemmType<Element, Layout>
    uint32_t COMPUTE_LENGTH_     // Length of a single computation
>
struct TileMuls {
    using Element = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
};
```

> `COMPUTE_LENGTH_` is a constant passed by the template parameter and is not controlled by runtime parameters. It is used for constant folding optimization.

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // UB destination tensor
    AscendC::LocalTensor<Element> srcTensor,    // UB Source tensor
    Element scalar,                              // Scalar
    uint32_t len                                 // Actual computation length
);
```

Execution process:
1. `SetMaskCount()` → `SetVectorMask<Element, COUNTER>(len)` Set mask.
2. `Muls<Element, false>(dst, src, scalar, MASK_PLACEHOLDER, 1, {} )`
3. `SetMaskNorm()` → `ResetMask()` Restore the mask.

## Examples

```cpp
#include "catlass/gemm/tile/tile_muls.hpp"

using namespace Catlass::Gemm;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 256;

using MulsOp = Tile::TileMuls<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

half scalar = 0.5_hf;
uint32_t len = 256;

AscendC::LocalTensor<half> srcUB;
AscendC::LocalTensor<half> dstUB;

MulsOp mulsOp;
mulsOp(dstUB, srcUB, scalar, len);

// Equivalent to: dstUB[i] = srcUB[i] * 0.5  for i in [0, len)
```
