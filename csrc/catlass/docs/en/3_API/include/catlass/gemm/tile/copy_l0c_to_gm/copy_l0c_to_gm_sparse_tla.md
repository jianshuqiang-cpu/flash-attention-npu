# CopyL0CToGmSparseTla

> [Code Location](../../../../../../../../include/catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp)

[TOC]

## Function

`CopyL0CToGmSparseTla` is a TLA-style data movement template specifically for Sparse GEMM L0C-to-GM movements. It moves the matrix multiply-accumulate result from L0C (CO1) to Global Memory (GM), with support for type conversions such as float32 → float16.

Unlike the regular [CopyL0CToGmTla](./tile_copy_tla.md), `CopyL0CToGmSparseTla` uses the `FixpipeParamsV220` parameter structure (instead of `FixpipeParams`) to control the movement dimensions, making it suitable for Sparse GEMM scenarios where sparse outputs are interleaved with dense outputs.

> **Restriction**: This template supports only the Atlas A2 architecture (`CATLASS_ARCH == 2201`) and is only in NO_QUANT mode.

## Template Prototype

```cpp
template <
    class ArchTag,                                          // Architecture tag
    class TensorSrc,                                        // Source tensor (L0C)
    class TensorDst,                                        // Destination tensor (GM)
    ScaleGranularity DEQUANT_GRANULARITY = NO_QUANT,        // Quantization mode
    class Enable = void                                     // SFINAE dispatch
>
struct CopyL0CToGmSparseTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag. Only `Arch::AtlasA2` is supported.|
| `TensorSrc` | Source tensor: `tla::Tensor<LocalTensor<ElementSrc>, Layout, Coord, CO1>`|
| `TensorDst` | Destination tensor: `tla::Tensor<GlobalTensor<ElementDst>, Layout, Coord, GM>`|
| `DEQUANT_GRANULARITY` | Quantization mode. Only `NO_QUANT` is supported.|
| `Enable` | SFINAE condition|

## Partial Specialization Implementation

### Atlas A2

| Source Layout| Destination Layout| Quantization Mode| SFINAE Condition| Description|
| :------ | :------ | :------ | :------ | :------ |
| L0C (zN)| RowMajor | NO_QUANT | `isRowMajor<LayoutDst>` | Fixpipe v220, CFG_ROW_MAJOR|

Movement dimensions (`nSize` and `mSize`) are controlled via the `FixpipeParamsV220` parameter structure, while the quantization precision is automatically deduced by `CopyL0CToGmQuantMode`.

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

- `dstTensor`: destination tensor in GM (RowMajor)
- `srcTensor`: source tensor in L0C (zN layout)

## Examples

### L0C → GM RowMajor (Atlas A2)

```cpp
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "tla/tensor.hpp"

using namespace Catlass::Gemm::Tile;

const uint32_t M = 256;
const uint32_t N = 256;

auto srcLayout = tla::MakeLayout<float, layout::zN>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<float> srcL0CTensor;
AscendC::GlobalTensor<half> dstGmTensor;

auto srcTensor = tla::MakeTensor(srcL0CTensor, srcLayout, Arch::PositionL0C{});
auto dstTensor = tla::MakeTensor(dstGmTensor, dstLayout, Arch::PositionGM{});

CopyL0CToGmSparseTla<Arch::AtlasA2, decltype(srcTensor), decltype(dstTensor)> sparseCopyOp;
sparseCopyOp(dstTensor, srcTensor);
```
