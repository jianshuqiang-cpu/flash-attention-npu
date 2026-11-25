# TileRowBroadcastMulTla

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_mul.hpp)

[TOC]

## 功能说明

`TileRowBroadcastMulTla` 实现 epilogue 阶段的 TLA 风格广播乘法操作。将 UB 上行向量 (1, n) 广播到 (m, n) 矩阵后与输入逐元素相乘。通过 `ubOut.layout()(ubOut.coord())` 计算偏移后调用 `AscendC::Mul`。

- 适用范围：所有架构（无架构特化）
- 风格：TLA

## 模板原型

```cpp
template <
    class ArchTag_,        // 架构标签
    class ElementCompute_, // 计算元素类型（直接传入，非 GemmType）
    class TileShape_       // Tile 形状
>
struct TileRowBroadcastMulTla;
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

通过 `ubOut.layout()(ubOut.coord())` 计算偏移后调用 `AscendC::Mul`。

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

TileRowBroadcastMulTla<Arch::AtlasA2, half, Shape<M, N>> op;
op(ubOut, ubIn0, ubIn1);
```
