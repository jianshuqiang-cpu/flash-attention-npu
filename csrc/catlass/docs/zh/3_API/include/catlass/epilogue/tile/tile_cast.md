# TileCast

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_cast.hpp)

[TOC]

## 功能说明

`TileCast` 实现 epilogue 阶段的类型转换操作，将 UB 上的源数据类型转换为目标数据类型。通过 AscendC 的 `Cast` 指令完成元素级类型转换，用于浮点量化、反量化等场景。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 类型转换由 `DstType_` 和 `SrcType_` 的 `Element` 成员决定

## 模板原型

```cpp
template <
    class ArchTag_,       // 架构标签
    class DstType_,       // 目标数据类型（含 Element）
    class SrcType_,       // 源数据类型（含 Element）
    class TileShape_      // Tile 形状（含 COUNT）
>
struct TileCast;
```

| 模板参数     | 说明                                                  |
| :----------- | :---------------------------------------------------- |
| `ArchTag_`   | 架构标签，通常为 `Arch::AtlasA2` 或 `Arch::Ascend950` |
| `DstType_`   | 目标数据类型，通过 `DstType_::Element` 获取元素类型   |
| `SrcType_`   | 源数据类型，通过 `SrcType_::Element` 获取元素类型     |
| `TileShape_` | Tile 形状类型，通过 `TileShape_::COUNT` 获取元素总数  |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementDst> const &ubOut,  // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementSrc> const &ubIn    // 源 UB LocalTensor
)
```

| 参数    | 说明                                       |
| :------ | :----------------------------------------- |
| `ubOut` | 目标 UB Tensor，类型为 `DstType_::Element` |
| `ubIn`  | 源 UB Tensor，类型为 `SrcType_::Element`   |

内部通过 `AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT)` 实现类型转换。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_cast.hpp"

using namespace Catlass::Epilogue::Tile;

using SrcType = Gemm::GemmType<int32_t, layout::RowMajor>;
using DstType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using CastOp = TileCast<Arch::AtlasA2, DstType, SrcType, TileShape>;

AscendC::LocalTensor<int32_t> srcTensor;
AscendC::LocalTensor<half> dstTensor;

CastOp castOp;
castOp(dstTensor, srcTensor);
```
