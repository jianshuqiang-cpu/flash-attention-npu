# TileOneBlkColumnBroadcastMul

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## 功能说明

`TileOneBlkColumnBroadcastMul` 实现 epilogue 阶段的列广播乘法操作。将形状 (m, 1) 的列向量在 block 内广播到 (m, n) 后与输入相乘。broadcast 粒度为一个 block（`BYTE_PER_BLK` 字节），即 column 上的每 1 个元素广播到 1 个完整的 block。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA

## 模板原型

```cpp
template <
    class ArchTag_,       // 架构标签
    class ComputeType_,   // 计算数据类型
    class TileShape_      // Tile 形状
>
struct TileOneBlkColumnBroadcastMul;
```

| 模板参数       | 说明                                       |
| :------------- | :----------------------------------------- |
| `ArchTag_`     | 架构标签                                   |
| `ComputeType_` | `Gemm::GemmType<ElementCompute, RowMajor>` |
| `TileShape_`   | Tile 形状，`Shape<ROW, COLUMN>`            |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,
    AscendC::LocalTensor<ElementCompute> const &ubIn0,
    AscendC::LocalTensor<ElementCompute> const &ubIn1     // (m, eleNumPerBlk) 形状
)
```

通过 `AscendC::Mul` + `BinaryRepeatParams`（`src1RepStride = 0`, `src1BlkStride = 1`）实现列广播。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
using TileShape = Shape<128, 256>;

using ColumnBroadcastMul = TileOneBlkColumnBroadcastMul<Arch::AtlasA2, ComputeType, TileShape>;

AscendC::LocalTensor<half> ubOut, ubIn0, ubIn1;

ColumnBroadcastMul op;
op(ubOut, ubIn0, ubIn1);
```
