# TileCopyTla (L0C → GM)

> [Code Location](../../../../../../../../include/catlass/gemm/tile/copy_l0c_to_gm.hpp)

[TOC]

## Function

`CopyL0CToGmTla` is the TLA-wrapped version of `CopyL0CToGm`. It is responsible for moving the matrix multiply-accumulate result from L0C (CO1) to Global Memory (GM), with support for type conversion, per-tensor/per-channel quantization and dequantization, and ReLU activation.

The key difference from the [non-TLA-version](./copy_l0c_to_gm.md) is that operands are encapsulated via `tla::Tensor`, and partial specializations are automatically dispatched via SFINAE based on the destination layout.

> **Note**: The structure name is `CopyL0CToGmTla`, which is different from the general `TileCopyTla` and is dedicated to the L0C-to-GM channel.

## Template Prototype

```cpp
template <
    class ArchTag,                                               // Architecture tag: Arch::AtlasA2 or Arch::Ascend950
    class TensorSrc,                                             // Source TLA tensor (L0C, CO1)
    class TensorDst,                                             // Destination TLA tensor (GM)
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,  // Quantization granularity
    bool ReluEnable = false,                                     // Whether to enable ReLU
    class Enable = void                                          // SFINAE dispatch
>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm.");
};
```

## Partial Specialization Implementation

Partial specialization is automatically dispatched through `Enable SFINAE`.

| Architecture| SFINAE Condition| Description|
| :------ | :------ | :------ |
| Atlas A2| `isRowMajor<LayoutDst>` | GM RowMajor, `AscendC::Fixpipe` + `CFG_ROW_MAJOR`|
| Atlas A2| `iszN<ElementDst, LayoutDst>` | GM zN, `AscendC::Fixpipe` + `CFG_NZ`|
| Ascend 950| `isRowMajor<LayoutDst>` + NO_QUANT | `AscendC::DataCopy` + `SetFixpipeNz2ndFlag` |
| Ascend 950| `iszN<ElementDst, LayoutDst>` + NO_QUANT | `AscendC::DataCopy`, zN hold|
| Ascend 950| `isRowMajor<LayoutDst>` + PER_TENSOR | `AscendC::Fixpipe` + `deqScalar` |
| Ascend 950| `isRowMajor<LayoutDst>` + PER_CHANNEL | `AscendC::Fixpipe` three-argument, taking the scale vector directly|

## APIs

### NO_QUANT/PER_TENSOR (basic overload)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,    // Destination tensor (GM, RowMajor, or zN)
    TensorSrc const &srcTensor,    // Source tensor (L0C, CO1)
    uint8_t unitFlag = 0           // Unit flag
);
```

### Ascend 950 Batch Movement Overload (NO_QUANT only)

```cpp
template <class TensorDst, class TensorSrc>
void operator()(
    TensorDst const &dstTensor,
    TensorSrc const &srcTensor,
    uint32_t l0Batch,              // Number of L0C batches
    uint32_t dstNdStride           // Destination ND stride
);
```

### PER_CHANNEL overload (including the scale tensor)

```cpp
template <class TensorDst, class TensorSrc, class TensorQuant>
void operator()(
    TensorDst const &dstTensor,        // Destination tensor.
    TensorSrc const &srcTensor,        // Source tensor
    TensorQuant const &quantTensor,    // Per-channel scale tensor
    uint8_t unitFlag = 0               // Unit flag
);
```

## Examples

### NO_QUANT RowMajor (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;
using namespace tla;

using ElementAccumulator = float;
using ElementDst = half;

const int M = 128;
const int N = 256;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### NO_QUANT zN (Atlas A2)

```cpp
auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<float, layout::zN>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### NO_QUANT RowMajor (Ascend 950)

```cpp
auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor)> copyOp;
copyOp(dstTensor, srcTensor);
```

### PER_TENSOR: int32 → half (Ascend 950)

```cpp
using ElementAccumulator = int32_t;
using ElementDst = half;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

using CopyOp = CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    ScaleGranularity::PER_TENSOR>;
CopyOp::Params params(0.5f);
CopyOp copyOp(params);
copyOp(dstTensor, srcTensor);
```

### PER_CHANNEL: float → int8 (Ascend 950)

```cpp
using ElementAccumulator = float;
using ElementDst = int8_t;

auto srcLayout = tla::MakeLayout<ElementAccumulator, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<ElementDst, layout::RowMajor>(M, N);

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

auto quantLayout = tla::MakeLayout<uint64_t, layout::VectorLayout>(N);
auto quantTensor = tla::MakeTensor(scaleData, quantLayout, Arch::PositionL1{});

CopyL0CToGmTla<Arch::Ascend950, decltype(srcTensor), decltype(dstTensor),
    ScaleGranularity::PER_CHANNEL> copyOp;
copyOp(dstTensor, srcTensor, quantTensor);
```
