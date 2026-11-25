# TileCopyGemvAiv

> [Code Location](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyGemvAiv` is an aggregation template that provides GEMV data movement child components for AI Vector chips. It implements the data path Global Memory (GM) ↔ Unified Buffer (UB) ↔ GM, leveraging GEMV-specific data movement components.

- Applicability: Atlas A2
- TileCopyGemvAiv does not directly execute operators. Instead, it exposes its child components as type members, allowing them to be accessed.

## Template Prototype

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileCopyGemvAiv;
```

| Parameter| Description|
| :------ | :------ |
| `ArchTag` | Architecture tag|
| `AType` | Matrix A type: `GemmType<ElementA, RowMajor/ColumnMajor>`|
| `XType` | Vector X type: `GemmType<ElementX, VectorLayout>`|
| `YType` | Vector Y type: `GemmType<ElementY, VectorLayout>`|
| `BiasType` | Bias type. The default value is `void`.|

## Member Types

| Member Type| Child Component| Description|
| :------ | :------ | :------ |
| `VecCopyGmToUb` | `Gemv::Tile::VecCopyGmToUB` | Vector X: GM → UB|
| `VecCopyUbToGm` | `Gemv::Tile::VecCopyUBToGm` | Vector Y: UB → GM (atomic add is optional)|
| `MatrixCopyGmToUb` | `Gemv::Tile::MatrixCopyGmToUB` | Matrix A: GM→UB|

## Examples

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using ElementX = half;
using ElementY = half;

using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementX, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementY, layout::VectorLayout>;

using Copy = TileCopyGemvAiv<Arch::AtlasA2, AType, XType, YType>;

// Child components
// typename Copy::VecCopyGmToUb
// typename Copy::VecCopyUbToGm
// typename Copy::MatrixCopyGmToUb
```
