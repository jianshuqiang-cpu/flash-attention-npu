# TileElemWiseAdd

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_add.hpp)

[TOC]

## 功能说明

`TileElemWiseAdd` 实现 epilogue 阶段的逐元素加法操作，对 UB 上的两个输入 Tensor 做 element-wise Add 并输出到目标 Tensor。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 通过 `AscendC::Add` 指令完成计算

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ComputeType_,       // 计算数据类型（含 Element）
    uint32_t COMPUTE_LENGTH_  // 计算长度（element 个数）
>
struct TileElemWiseAdd;
```

| 模板参数          | 说明                                                    |
| :---------------- | :------------------------------------------------------ |
| `ArchTag_`        | 架构标签                                                |
| `ComputeType_`    | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `COMPUTE_LENGTH_` | 计算长度，即需要参与计算的元素个数                      |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> const &ubOut,   // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementCompute> const &ubIn0,   // 源 UB LocalTensor 0
    AscendC::LocalTensor<ElementCompute> const &ubIn1    // 源 UB LocalTensor 1
)
```

| 参数    | 说明               |
| :------ | :----------------- |
| `ubOut` | 目标 UB Tensor     |
| `ubIn0` | 第一个源 UB Tensor |
| `ubIn1` | 第二个源 UB Tensor |

内部实现：`AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH)`

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using AddOp = TileElemWiseAdd<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> ubOut;
AscendC::LocalTensor<half> ubIn0;
AscendC::LocalTensor<half> ubIn1;

AddOp addOp;
addOp(ubOut, ubIn0, ubIn1);
```
