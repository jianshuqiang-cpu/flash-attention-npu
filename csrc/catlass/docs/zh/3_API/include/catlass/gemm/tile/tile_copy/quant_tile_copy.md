# QuantTileCopy

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`QuantTileCopy` 继承自 [TileCopy](./tile_copy.md)，**重写** `CopyL0CToGm` 为指定量化粒度（默认 `PER_TENSOR`），并**新增** `CopyGmToL1Scale` 和 `CopyL1ToFP` 两条搬运通道，用于量化 Scale 数据的搬入和 FixPipe 旁路。

适用于量化推理/训练场景：GM 上的 Scale 数据通过 `CopyGmToL1Scale` 搬入 L1，再通过 `CopyL1ToFP` 旁路到 FixPipe，在 L0C→GM 搬运时随路量化。

## 引用的 Tile 组件

| 成员别名                     | 来源                                                                             |
| :--------------------------- | :------------------------------------------------------------------------------- |
| `CopyGmToL1A` ~ `CopyL1ToBT` | 继承自 `TileCopy<ArchTag, AType, BType, CType, BiasType>`                        |
| `CopyL0CToGm`（重写）        | `CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>`            |
| `CopyGmToL1Scale`（新增）    | `CopyGmToL1<ArchTag, uint64_t VectorLayout GM, uint64_t VectorLayout A1>`        |
| `CopyL1ToFP`（新增）         | `CopyL1ToFP<ArchTag, uint64_t VectorLayout A1, uint64_t VectorLayout C2PIPE2GM>` |

## 模板原型

```cpp
template <
    class ArchTag,                                                   // 架构标签
    class AType,                                                     // A 矩阵 GmType
    class BType,                                                     // B 矩阵 GmType
    class CType,                                                     // C 矩阵 GmType
    class BiasType = void,                                           // Bias GmType（可选）
    ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR      // 量化粒度
>
struct QuantTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType>;
```

## 重写与新增的成员

```cpp
// 重写：量化 L0C→GM
using CopyL0CToGm = CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;

// 新增：Scale GM→L1
using CopyGmToL1Scale = CopyGmToL1<ArchTag,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>>;

// 新增：L1→FP（FixPipe）
using CopyL1ToFP = CopyL1ToFP<ArchTag,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
    Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>>;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using TileCopy_ = Tile::QuantTileCopy<Arch::AtlasA2, AType, BType, CType,
    void, ScaleGranularity::PER_TENSOR>;

typename TileCopy_::CopyGmToL1A     copyGmToL1A;
typename TileCopy_::CopyGmToL1Scale copyGmToL1Scale;   // Scale 搬入
typename TileCopy_::CopyL1ToFP      copyL1ToFP;         // Scale 旁路
typename TileCopy_::CopyL0CToGm     copyL0CToGm;        // 随路量化写回
```
