# CopyUb2L1Tla

> [Code Location](../../../../../../../include/catlass/epilogue/tile/copy_ub_to_l1_tla.hpp)

[TOC]

## Function

`CopyUb2L1Tla` implements zN-formatted data movement (TLA style) from Unified Buffer to L1 in the epilogue stage. It moves zN-formatted data from Unified Buffer to L1 while preserving the zN format.

- Applicability: only `Arch::Ascend950` (conditional compilation `CATLASS_ARCH == 3510`)
- Style: TLA (operands are encapsulated using `tla::Tensor`.)
- Layout requirements: source Unified Buffer in zN format (unaligned), and the destination L1 in zN format (aligned)
- Data movement implemented using `AscendC::DataCopy`

## Template Prototype

```cpp
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct CopyUb2L1Tla;
```

At the underlying layer,  SFINAE matches `iszNUnAlign<ElementSrc, LayoutSrc>` && `iszN<ElementDst, LayoutDst>`, with `TPosition::VECCALC` → `TPosition::A1`.

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag, specialization only for Ascend 950|
| `TensorSrc` | Source TLA tensor, Unified Buffer location, zN unaligned layout|
| `TensorDst` | Destination TLA tensor, L1 location, zN aligned layout|

## APIs

```cpp
template <class TensorDst, class TensorSrc>
void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
```

| Parameter| Description|
| :------ | :------ |
| `dstTensor` | Destination L1 TLA tensor (zN aligned layout, TPosition::A1)|
| `srcTensor` | Source Unified Buffer TLA tensor (zN unaligned layout, TPosition::VECCALC)|

Internally, `AscendC::DataCopy(dstData[dstOffset], srcData[srcOffset], dataCopyParams)` is used, and data is moved by segment according to the zN format.

## Examples

```cpp
#include "catlass/epilogue/tile/copy_ub_to_l1_tla.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128;
constexpr uint32_t N = 256;

// Source Unified Buffer: zN unaligned
auto srcLayout = tla::MakeLayout<half, layout::zNUnAlign>(M, N);
auto dstLayout = tla::MakeLayout<half, layout::zN>(M, N);

AscendC::LocalTensor<half> ubTensor;
AscendC::LocalTensor<half> l1Tensor;

auto srcTlaTensor = tla::MakeTensor(ubTensor, srcLayout, Arch::PositionUB{});
auto dstTlaTensor = tla::MakeTensor(l1Tensor, dstLayout, Arch::PositionL1{});

CopyUb2L1Tla<Arch::Ascend950, decltype(srcTlaTensor), decltype(dstTlaTensor)> copyOp;
copyOp(dstTlaTensor, srcTlaTensor);
```
