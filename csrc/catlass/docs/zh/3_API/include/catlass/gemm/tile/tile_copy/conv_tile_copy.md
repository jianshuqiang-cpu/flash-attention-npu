# ConvTileCopy

> [代码位置](../../../../../../../../include/catlass/gemm/tile/tile_copy.hpp)

[TOC]

## 功能说明

`ConvTileCopy` 是卷积（Conv）场景的搬运模板集合，结构上与 [TileCopy](./tile_copy.md) 完全相同，区别仅在于模板参数命名和用途——专门服务于 Convolution 的 Im2Col + GEMM 流程。

BiasType 为必选参数（非默认 `void`），因为 Conv 通常包含 Bias。

## 引用的 Tile 组件

| 成员别名         | 引用的底层模板                                    | 说明          |
| :--------------- | :------------------------------------------------ | :------------ |
| `CopyGmToL1A`    | `CopyGmToL1<ArchTag, AType>`                      | A 矩阵 GM→L1  |
| `CopyGmToL1B`    | `CopyGmToL1<ArchTag, BType>`                      | B 矩阵 GM→L1  |
| `CopyL1ToL0A`    | `CopyL1ToL0A<ArchTag, L1AType>`                   | A 矩阵 L1→L0A |
| `CopyL1ToL0B`    | `CopyL1ToL0B<ArchTag, L1BType>`                   | B 矩阵 L1→L0B |
| `CopyL0CToGm`    | `CopyL0CToGm<ArchTag, ElementAccumulator, CType>` | L0C→GM        |
| `CopyGmToL1Bias` | `CopyGmToL1<ArchTag, ...>`                        | Bias GM→L1    |
| `CopyL1ToBT`     | `CopyL1ToBT<ArchTag, ...>`                        | Bias L1→BT    |

## 模板原型

```cpp
template <
    class ArchTag,            // 架构标签
    class AType,              // A 矩阵 GmType
    class BType,              // B 矩阵 GmType
    class CType,              // C 矩阵 GmType
    class BiasType            // Bias GmType（必选）
>
struct ConvTileCopy;
```

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_copy.hpp"

using namespace Catlass::Gemm;

using AType = Gemm::GemmType<half, layout::RowMajor>;
using BType = Gemm::GemmType<half, layout::ColumnMajor>;
using CType = Gemm::GemmType<half, layout::NDC1HWC0>;
using BiasType = Gemm::GemmType<half, layout::VectorLayout>;

using TileCopy_ = Tile::ConvTileCopy<Arch::AtlasA2, AType, BType, CType, BiasType>;

typename TileCopy_::CopyGmToL1A   copyGmToL1A;
typename TileCopy_::CopyGmToL1B   copyGmToL1B;
typename TileCopy_::CopyGmToL1Bias copyGmToL1Bias;
typename TileCopy_::CopyL1ToBT    copyL1ToBT;
typename TileCopy_::CopyL0CToGm   copyL0CToGm;
```
