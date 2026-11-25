# TileElemWiseMuls

> [代码位置](../../../../../../../include/catlass/epilogue/tile/tile_elemwise_muls.hpp)

[TOC]

## 功能说明

`TileElemWiseMuls` 实现 epilogue 阶段的逐元素乘以标量操作，对 UB 上的输入 Tensor 每个元素乘以一个标量值并输出。

- 适用范围：所有架构（无架构特化）
- 风格：非 TLA，直接操作 `AscendC::LocalTensor`
- 通过 `AscendC::Muls` 指令完成计算

## 模板原型

```cpp
template <
    class ArchTag_,           // 架构标签
    class ComputeType_,       // 计算数据类型（含 Element）
    uint32_t COMPUTE_LENGTH_  // 计算长度（element 个数）
>
struct TileElemWiseMuls;
```

| 模板参数          | 说明                                                    |
| :---------------- | :------------------------------------------------------ |
| `ArchTag_`        | 架构标签                                                |
| `ComputeType_`    | 计算数据类型，通过 `ComputeType_::Element` 获取元素类型 |
| `COMPUTE_LENGTH_` | 计算长度                                                |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<ElementCompute> dstLocal,     // 目的 UB LocalTensor
    AscendC::LocalTensor<ElementCompute> srcTensor,    // 源 UB LocalTensor
    ElementCompute scalar                              // 标量值
)
```

| 参数        | 说明                                                  |
| :---------- | :---------------------------------------------------- |
| `dstLocal`  | 目标 UB Tensor（被覆盖），存放 `src[i] * scalar` 结果 |
| `srcTensor` | 源 UB Tensor                                          |
| `scalar`    | 标量乘数                                              |

内部实现：`AscendC::Muls(dstLocal, srcTensor, scalar, COMPUTE_LENGTH)`

## 调用示例

```cpp
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"

using namespace Catlass::Epilogue::Tile;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 128 * 256;

using MulsOp = TileElemWiseMuls<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

AscendC::LocalTensor<half> dstTensor;
AscendC::LocalTensor<half> srcTensor;
half scalar = 0.5_h;

MulsOp mulsOp;
mulsOp(dstTensor, srcTensor, scalar);
```
