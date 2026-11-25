# TileCopyGemvAic

> [Code Location](../../../../../../../../include/catlass/gemv/tile/tile_copy.hpp)

[TOC]

## Function

`TileCopyGemvAic` is an aggregation template that provides GEMV data movement child components for AI Core chips. It implements the data path Global Memory (GM) → L1 → L0A/L0B → L0C → GM, reusing the existing GEMM data movement components. The types for the intermediate L1, L0A, and L0B are automatically deduced using `Gemv::helper::L1AndL0TypeSelectorGemv`.

- Applicability: Ascend 950
- TileCopyGemvAic does not directly execute operators. Instead, it exposes its child components as type members, allowing them to be accessed.

## Template Prototype

```cpp
template <class ArchTag, class AType, class XType, class YType, class BiasType = void>
struct TileCopyGemvAic;
```

## Member Types

| Member Type| Child Component| Description|
| :------ | :------ | :------ |
| `CopyGmToL1A` | `Gemm::Tile::CopyGmToL1<Arch, XType, L1XType>` | Vector X: GM→L1|
| `CopyGmToL1B` | `Gemm::Tile::CopyGmToL1<Arch, AType, L1AType>` | Matrix A: GM→L1|
| `CopyL1ToL0A` | `Gemm::Tile::CopyL1ToL0A<Arch, L1XType, L0AType>` | L1→L0A |
| `CopyL1ToL0B` | `Gemm::Tile::CopyL1ToL0B<Arch, L1AType, L0BType>` | L1→L0B |
| `CopyL0CToGm` | `Gemm::Tile::CopyL0CToGm<Arch, ElementAccumulator, YType>` | L0C→GM |

## Examples

```cpp
#include "catlass/gemv/tile/tile_copy.hpp"

using namespace Catlass::Gemv::Tile;

using ElementA = half;
using AType = Gemm::GemmType<ElementA, layout::RowMajor>;
using XType = Gemm::GemmType<ElementA, layout::VectorLayout>;
using YType = Gemm::GemmType<ElementA, layout::VectorLayout>;

using Copy = TileCopyGemvAic<Arch::Ascend950, AType, XType, YType>;

// Child components
// typename Copy::CopyGmToL1A
// typename Copy::CopyGmToL1B
// typename Copy::CopyL1ToL0A
// typename Copy::CopyL1ToL0B
// typename Copy::CopyL0CToGm
```
