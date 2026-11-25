# TileRowBroadcastMul

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## 功能说明

`TileRowBroadcastMul` 实现 epilogue 阶段的广播乘法操作。将 UB 上行向量（1, n）广播到 (m, n) 矩阵后与输入逐元素相乘。通过 `AscendC::Mul` + `BinaryRepeatParams`（`src1RepStride = 0`）实现行广播。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA

## 模板原型

```cpp
template <
    class ArchTag_,       // 架构标签
    class ComputeType_,   // 计算数据类型
    class TileShape_      // Tile 形状（含 ROW 和 COLUMN）
>
struct TileRowBroadcastMul;
```

| 模板参数       | 说明                                       |
| :------------- | :----------------------------------------- |
| `ArchTag_`     | 架构标签                                   |
| `ComputeType_` | `Gemm::GemmType<ElementCompute, RowMajor>` |
| `TileShape_`   | Tile 形状，`Shape<ROW, COLUMN>`            |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,    // 目的 UB
    AscendC::LocalTensor<ElementCompute> const &ubIn0,    // 源 UB 0（m, n）
    AscendC::LocalTensor<ElementCompute> const &ubIn1     // 源 UB 1（1, n）行向量
)
```

通过 `AscendC::Mul` + `BinaryRepeatParams`（`src1RepStride = 0`）实现行广播。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;
using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using BroadcastMul = TileRowBroadcastMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut, ubIn0, ubIn1;

BroadcastMul broadcastMul;
broadcastMul(ubOut, ubIn0, ubIn1);
```
