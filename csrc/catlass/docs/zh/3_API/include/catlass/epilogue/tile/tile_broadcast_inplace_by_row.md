# TileBroadcastInplaceByRow

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_broadcast_inplace_by_row.hpp)

[TOC]

## 功能说明

`TileBroadcastInplaceByRow` 实现 epilogue 阶段的行广播原地拷贝操作。将 UB 上 (m, n) Tile 中第一行元素广播到下面的所有行，结果覆盖原地。常用于将行向量（per-token scale/zero点）广播到整个矩阵。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 通过 `AscendC::Copy` + `CopyRepeatParams` 实现广播

## 模板原型

```cpp
template <
    class ArchTag_,         // 架构标签
    class ComputeType_,     // 计算数据类型（含 Element）
    class TileShape_        // Tile 形状类型（含 ROW 和 COLUMN）
>
struct TileBroadcastInplaceByRow;
```

| 模板参数       | 说明                                                             |
| :------------- | :--------------------------------------------------------------- |
| `ArchTag_`     | 架构标签                                                         |
| `ComputeType_` | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型          |
| `TileShape_`   | Tile 形状，`TileShape_::ROW` 为行数，`TileShape_::COLUMN` 为列数 |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubInOut   // UB 输入输出 Tensor（原地）
)
```

| 参数      | 说明                                                    |
| :-------- | :------------------------------------------------------ |
| `ubInOut` | UB Tensor，输入为 (m, n) 矩阵，输出第一行被广播到所有行 |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_inplace_by_row.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastOp = TileBroadcastInplaceByRow<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubInOut;

BroadcastOp broadcastOp;
broadcastOp(ubInOut);
```
