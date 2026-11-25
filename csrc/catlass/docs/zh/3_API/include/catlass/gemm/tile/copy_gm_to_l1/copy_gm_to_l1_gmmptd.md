# CopyGmToL1GMMPTD

> [代码位置](../../../../../../../../include/catlass/gemm/tile/copy_gm_to_l1.hpp)

[TOC]

## 功能说明

`CopyGmToL1GMMPTD` 是非 TLA 风格的 GM 到 L1 数据搬运模板，专为 GMM（Group Matrix Multiplication）PTD（Permute-Transpose-DataCopy）场景设计。该模板在 `Nd2Nz` 搬运基础上增加了对单行矩阵的特殊优化（使用 strided DataCopy），并支持手动指定搬运 stride 的扩展调用接口。

支持 `Arch::AtlasA2` 和 `Arch::Ascend950` 两种架构。

## 模板原型

```cpp
template <
    class ArchTag,          // 架构标签
    class GmType,           // GM 上操作数的 Gemm 类型
    class L1Type = void     // L1 上操作数的 Gemm 类型（默认 void）
>
struct CopyGmToL1GMMPTD
```

### 模板参数说明

| 参数      | 说明                                                |
| :-------- | :-------------------------------------------------- |
| `ArchTag` | 架构标签，可选 `Arch::AtlasA2` 或 `Arch::Ascend950` |
| `GmType`  | GM 上源操作数的 Gemm 类型                           |
| `L1Type`  | L1 上目的操作数的 Gemm 类型，默认为 `void`          |

## 偏特化实现

| ArchTag           | GmType                        | 目的 Layout | 说明                      |
| :---------------- | :---------------------------- | :---------- | :------------------------ |
| `Arch::AtlasA2`   | `GemmType<Element, RowMajor>` | `zN`        | RowMajor → zN，含单行优化 |
| `Arch::Ascend950` | `GemmType<Element, RowMajor>` | `zN`        | RowMajor → zN，含单行优化 |

## 调用接口

### 基础调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc                        // 源操作数 layout
)
```

| 参数        | 说明                     |
| :---------- | :----------------------- |
| `dstTensor` | 目的 L1 LocalTensor      |
| `srcTensor` | 源 GM GlobalTensor       |
| `layoutDst` | 目的操作数的 layout 描述 |
| `layoutSrc` | 源操作数的 layout 描述   |

### 扩展调用接口（手动指定 stride）

```cpp
void operator()(
    AscendC::LocalTensor<Element> const &dstTensor,   // 目的操作数 LocalTensor
    AscendC::GlobalTensor<Element> const &srcTensor,  // 源操作数 GlobalTensor
    LayoutDst const &layoutDst,                       // 目的操作数 layout
    LayoutSrc const &layoutSrc,                       // 源操作数 layout
    uint32_t ndNum,                                   // ND 矩阵数量
    uint32_t srcNdMatrixStride,                       // 源 ND 矩阵间 stride
    uint32_t dstNzNStride,                            // 目的 n 方向 stride
    uint32_t dstNzMatrixStride,                       // 目的矩阵间 stride
    uint32_t dstNzC0Stride                            // 目的 C0 方向 stride
)
```

| 参数                | 说明                                            |
| :------------------ | :---------------------------------------------- |
| `ndNum`             | 连续搬运的 ND 矩阵数量                          |
| `srcNdMatrixStride` | 源端相邻 ND 矩阵间的 stride                     |
| `dstNzNStride`      | 目的端 n 方向的 stride（覆盖 layout 默认值）    |
| `dstNzMatrixStride` | 目的端相邻矩阵间的 stride（覆盖 layout 默认值） |
| `dstNzC0Stride`     | 目的端 C0 方向的 stride（覆盖 layout 默认值）   |

## 调用示例

```cpp
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"

using namespace Catlass::Gemm::Tile;

using LayoutTagSrc = layout::RowMajor;
using LayoutTagDst = layout::zN;
using ElementDst = half;

// GMM PTD 场景只需指定 GmType（L1Type 默认为 void，由偏特化自动推导）
using GmType = Gemm::GemmType<ElementDst, LayoutTagSrc>;

uint32_t row = 256;
uint32_t col = 256;

// 构造 layout
auto layoutSrc = LayoutTagSrc::MakeLayout<ElementDst>(row, col);
auto layoutDst = LayoutTagDst::MakeLayout<ElementDst>(row, col);

AscendC::GlobalTensor<ElementDst> srcTensor;
AscendC::LocalTensor<ElementDst> dstTensor;

// 基础调用
using CopyOp = CopyGmToL1GMMPTD<Arch::AtlasA2, GmType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, layoutDst, layoutSrc);

// 扩展调用：手动指定 stride（多矩阵搬运场景）
// copyOp(dstTensor, srcTensor, layoutDst, layoutSrc,
//        ndNum, srcNdMatrixStride, dstNzNStride, dstNzMatrixStride, dstNzC0Stride);
```
