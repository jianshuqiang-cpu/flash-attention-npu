# TileVmad

> [Code Location](../../../../../../../include/catlass/gemv/tile/tile_vmad.hpp)

[TOC]

## Function

`TileVmad` is a template that implements the vector-matrix multiply-accumulate operation (`Y += A * X`) for GEMV scenarios on Atlas A2. It computes the dot product of each row of matrix A (of size m × n) with vector X (of size n), and accumulates the result into vector Y (of size m).

- Applicability: Atlas A2
- Two implementations: RowMajor (MulAddDst + WholeReduceSum) and ColumnMajor (per-column Axpy approach)

## Template Prototype

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileVmad;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `AType` | Matrix A type: `GemmType<ElementA, RowMajor/ColumnMajor>`|
| `XType` | Vector X type: `GemmType<ElementX, VectorLayout>`|
| `YType` | Vector Y type: `GemmType<ElementY, VectorLayout>`|
| `BiasType` | Bias type. The default value is `void`.|

## Partial Specialization Implementation

| Architecture| AType | Implementation Strategy| Special Version|
| :------ | :------ | :------ | :------ |
| Atlas A2| `RowMajor` | `Duplicate`→`MulAddDst`→`WholeReduceSum`→`Cast`→`Add` | float version: `Mul` + `MulAddDst`|
| Atlas A2| `ColumnMajor` | `Duplicate` → scalar shuffle → per-column `Axpy` → `Cast` → `Add`| float version synchronization|

**RowMajor implementation process:**
1. `Duplicate` initializes the accumulator buffer temp to zero.
2. Blocked `MulAddDst` computes `A[i:*n] * X[i:*n]` and accumulates the results into temp.
3. `WholeReduceSum` reduces temp into a column vector.
4. `Cast` converts the result back to the ElementA data type.
5. `Add` accumulates the final result into Y.

**ColumnMajor implementation process:**
1. `Duplicate` initializes temp to zero.
2. `SetFlag and WaitFlag` are used to load the scalar values from vector X into scalar registers.
3. Per-column `Axpy` updates `temp += A[:,i] * pix[i]` for each column.
4. `Cast` and `Add` accumulate the final result into Y.

## APIs

```cpp
void operator()(
    AscendC::LocalTensor<ElementY> dstTensor,           // Y vector (UB, including the accumulation result)
    AscendC::LocalTensor<ElementX> srcTensor_v,         // X vector (UB)
    AscendC::LocalTensor<ElementA> srcTensor_m,         // A matrix (UB)
    AscendC::LocalTensor<ElementAccumulator> temp,      // Temporary buffer (UB)
    LayoutDst const &layoutDst,                         // Actual layout of the A matrix
    LayoutSrc const &layoutSrc                          // Round layout of the A matrix
)
```

## Examples

### RowMajor

```cpp
#include "catlass/gemv/tile/tile_vmad.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using ElementX = half;
using ElementY = half;

using LayoutTagSrc = layout::RowMajor;

uint32_t m = 64, n = 128;

auto layoutSrc = LayoutTagSrc::MakeLayout<ElementA>(m, n);
auto layoutDst = LayoutTagSrc::MakeLayout<ElementA>(m, n);

AscendC::LocalTensor<ElementY> dstTensor;
AscendC::LocalTensor<ElementX> srcTensor_v;
AscendC::LocalTensor<ElementA> srcTensor_m;
AscendC::LocalTensor<float> temp;

using AType = Gemm::GemmType<ElementA, LayoutTagSrc>;
using XType = Gemm::GemmType<ElementX, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementY, layout::VectorLayout>;

using VmadOp = TileVmad<Arch::AtlasA2, AType, XType, YType>;
VmadOp vmad;
vmad(dstTensor, srcTensor_v, srcTensor_m, temp, layoutDst, layoutSrc);
```
