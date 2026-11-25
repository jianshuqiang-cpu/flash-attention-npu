# TileRowBroadcastAdd

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_broadcast_add.hpp)

[TOC]

## 功能说明

`TileRowBroadcastAdd` 实现 epilogue 阶段的行广播加法操作。将一个 (1, n) 的行向量广播到 (m, n) 并与另一个 (m, n) 矩阵逐元素相加，输出到目标 Tensor。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 通过 `AscendC::Add` + `BinaryRepeatParams` 实现广播加法

## 模板原型

```cpp
template <
    class ArchTag_,       // 架构标签
    class ComputeType_,   // 计算数据类型（含 Element）
    class TileShape_      // Tile 形状类型（含 ROW 和 COLUMN）
>
struct TileRowBroadcastAdd;
```

| 模板参数       | 说明                                                    |
| :------------- | :------------------------------------------------------ |
| `ArchTag_`     | 架构标签                                                |
| `ComputeType_` | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `TileShape_`   | Tile 形状，`TileShape_::COLUMN` 用于计算分块参数        |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,           // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementCompute> const &ubIn0,           // 源 UB LocalTensor 0（m, n）
    AscendC::LocalTensor<ElementCompute> const &ubIn1,           // 源 UB LocalTensor 1（1, n）行向量
    MatrixCoord const &actualTileShape                           // 实际 Tile 形状 (m, n)
)
```

| 参数              | 说明                                            |
| :---------------- | :---------------------------------------------- |
| `ubOut`           | 目标 UB Tensor，存放 `ubIn0[i] + ubIn1` 结果    |
| `ubIn0`           | (m, n) 形状的 UB Tensor                         |
| `ubIn1`           | (1, n) 形状的行向量，广播后与 ubIn0 逐元素相加  |
| `actualTileShape` | `MatrixCoord{rows, cols}`，实际处理的 tile 维度 |

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_add.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastAddOp = TileRowBroadcastAdd<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;
MatrixCoord actualTileShape(128, 256);

BroadcastAddOp broadcastAddOp;
broadcastAddOp(ubOut, ubIn0, ubIn1, actualTileShape);
```
