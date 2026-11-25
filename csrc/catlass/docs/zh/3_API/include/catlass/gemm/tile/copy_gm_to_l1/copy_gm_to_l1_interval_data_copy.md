# CopyGmToL1IntervalDataCopy

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1IntervalDataCopy` 是非 TLA 风格的 GM 到 L1 数据搬运模板。与 `CopyGmToL1` 使用 `Nd2Nz` 指令不同，该模板使用标准的 strided `DataCopy` 接口逐行搬运数据，在数据块形状为"矮宽"（short and wide）或"高窄"（tall and narrow）时可能获得更高的搬运效率。

当前仅支持 `Arch::AtlasA2` 架构和 `half` 数据类型。

## 模板原型

```cpp
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM 上操作数的 Gemm 类型
    class L1Type = void     // L1 上操作数的 Gemm 类型（默认 void）
>
struct CopyGmToL1IntervalDataCopy
```

### 模板参数说明

| 参数      | 说明                                       |
| :-------- | :----------------------------------------- |
| `ArchTag` | 架构标签，当前仅支持 `Arch::AtlasA2`       |
| `GmType`  | GM 上源操作数的 Gemm 类型                  |
| `L1Type`  | L1 上目的操作数的 Gemm 类型，默认为 `void` |

## 偏特化实现

所有偏特化仅适用于 `Arch::AtlasA2`，数据类型固定为 `half`。

| GmType                               | 目的 Layout | 说明                                  |
| :----------------------------------- | :---------- | :------------------------------------ |
| `GemmType<half, RowMajor>`           | `zN`        | RowMajor → zN，逐行间隔搬运           |
| `GemmType<half, PaddingRowMajor>`    | `zN`        | PaddingRowMajor → zN，逐行间隔搬运    |
| `GemmType<half, ColumnMajor>`        | `nZ`        | ColumnMajor → nZ，逐列间隔搬运        |
| `GemmType<half, PaddingColumnMajor>` | `nZ`        | PaddingColumnMajor → nZ，逐列间隔搬运 |

## 调用接口

所有偏特化使用统一的调用接口：

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc                        // 源操作数 layout
)
```

| 参数        | 说明                                   |
| :---------- | :------------------------------------- |
| `dstTensor` | 目的 L1 LocalTensor，元素类型为 `half` |
| `srcTensor` | 源 GM GlobalTensor，元素类型为 `half`  |
| `layoutDst` | 目的操作数的 layout 描述               |
| `layoutSrc` | 源操作数的 layout 描述                 |

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;

// CopyGmToL1IntervalDataCopy 当前仅支持 half 类型和 AtlasA2 架构
using GmType = Gemm::GemmType<half, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<half>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<half>(row, col);

AscendC::GlobalTensor<half> srcTensor;
AscendC::LocalTensor<half> dstTensor;

// 使用 strided DataCopy 逐行搬运，适用于矮宽/高窄数据块
using CopyOp = CopyGmToL1IntervalDataCopy<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);
```
