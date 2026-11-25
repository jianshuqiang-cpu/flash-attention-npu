# TileBroadcastOneBlkTla

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## 功能说明

`TileBroadcastOneBlkTla` 实现 TLA 风格的 one-block 广播操作。与 `TileBroadcastOneBlk` 功能相同，通过 `tla::Tensor` 封装接口。

- 适用范围：所有架构（无架构特化）
- 风格：TLA

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ElementCompute_,    // 计算元素类型（直接传入）
    uint32_t COMPUTE_LENGTH_  // 计算长度
>
struct TileBroadcastOneBlkTla;
```

| 模板参数          | 说明                    |
| :---------------- | :---------------------- |
| `ArchTag_`        | 架构标签                |
| `ElementCompute_` | 计算元素类型，如 `half` |
| `COMPUTE_LENGTH_` | 需要广播的元素总数      |

## 调用接口

```cpp
template <class TensorUbOut, class TensorUbIn>
void operator()(TensorUbOut &ubOut, TensorUbIn &ubIn)
```

通过 `ubOut.layout()(ubOut.coord())` 计算偏移后调用 `AscendC::Brcb`。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"

using namespace Catlass::Epilogue::Tile;
constexpr uint32_t COMPUTE_LENGTH = 256;

auto layoutOut = tla::MakeLayout<half, layout::RowMajor>(COMPUTE_LENGTH, 32);
auto layoutIn = tla::MakeLayout<half, layout::VectorLayout>(COMPUTE_LENGTH, 1);

AscendC::LocalTensor<half> ubOutData, ubInData;
auto ubOut = tla::MakeTensor(ubOutData, layoutOut, Arch::PositionUB{});
auto ubIn = tla::MakeTensor(ubInData, layoutIn, Arch::PositionUB{});

TileBroadcastOneBlkTla<Arch::AtlasA2, half, COMPUTE_LENGTH> op;
op(ubOut, ubIn);
```
