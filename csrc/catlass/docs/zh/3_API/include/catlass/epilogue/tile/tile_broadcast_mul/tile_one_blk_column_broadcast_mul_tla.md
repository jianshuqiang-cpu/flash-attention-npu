# TileOneBlkColumnBroadcastMulTla

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## 功能说明

`TileOneBlkColumnBroadcastMulTla` 实现 TLA 风格的一 block 列广播乘法操作。与 `TileOneBlkColumnBroadcastMul` 功能相同，通过 `tla::Tensor` 封装接口。

- 适用范围：所有架构（无架构特化）
- 风格：TLA

## 模板原型

```cpp
template <
    class ArchTag_,        // 架构标签
    class ElementCompute_, // 计算元素类型
    class TileShape_       // Tile 形状
>
struct TileOneBlkColumnBroadcastMulTla;
```

| 模板参数          | 说明                            |
| :---------------- | :------------------------------ |
| `ArchTag_`        | 架构标签                        |
| `ElementCompute_` | 计算元素类型，如 `half`         |
| `TileShape_`      | Tile 形状，`Shape<ROW, COLUMN>` |

## 调用接口

```cpp
template <class TensorUbOut, class TensorUbIn0, class TensorUbIn1>
void operator()(TensorUbOut const &ubOut, TensorUbIn0 const &ubIn0, TensorUbIn1 const &ubIn1)
```

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"

using namespace Catlass::Epilogue::Tile;

constexpr uint32_t M = 128, N = 256;

auto layout = tla::MakeLayout<half, layout::RowMajor>(M, N);

AscendC::LocalTensor<half> ubOutData, ubIn0Data, ubIn1Data;
auto ubOut = tla::MakeTensor(ubOutData, layout, Arch::PositionUB{});
auto ubIn0 = tla::MakeTensor(ubIn0Data, layout, Arch::PositionUB{});
auto ubIn1 = tla::MakeTensor(ubIn1Data, layout, Arch::PositionUB{});

TileOneBlkColumnBroadcastMulTla<Arch::AtlasA2, half, Shape<M, N>> op;
op(ubOut, ubIn0, ubIn1);
```
