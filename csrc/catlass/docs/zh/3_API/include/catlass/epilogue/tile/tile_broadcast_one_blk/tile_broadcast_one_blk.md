# TileBroadcastOneBlk

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_broadcast_one_blk.hpp)

[TOC]

## 功能说明

`TileBroadcastOneBlk` 实现 epilogue 阶段的 one-block 广播操作。将 UB 上的单个元素广播到整个 block（32B），常用于将 scalar scale/zero 点广播后参与向量计算。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ComputeType_,       // 计算数据类型（含 Element）
    uint32_t COMPUTE_LENGTH_  // 计算长度
>
struct TileBroadcastOneBlk;
```

| 模板参数          | 说明                                                    |
| :---------------- | :------------------------------------------------------ |
| `ArchTag_`        | 架构标签                                                |
| `ComputeType_`    | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `COMPUTE_LENGTH_` | 需要广播的元素总数                                      |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,    // 目的 UB（广播后）
    AscendC::LocalTensor<ElementCompute> const &ubIn      // 源 UB（COMPUTE_LENGTH 个元素）
)
```

内部通过 `AscendC::Brcb`（BroadCast to Block）+ `BrcbRepeatParams` 将每个元素广播到完整 block。

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 256;

using BroadcastOp = TileBroadcastOneBlk<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> ubOut, ubIn;

BroadcastOp broadcastOp;
broadcastOp(ubOut, ubIn);
```
