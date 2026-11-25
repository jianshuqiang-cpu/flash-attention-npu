# TileVmuls

> [代码位置](../../../../../../../include/catlass/gemv/tile/tile_vmuls.hpp)

[TOC]

## 功能说明

`TileVmuls` 实现 GEMV 场景下向量乘以标量的操作。通过 `AscendC::Muls` 对 UB 上指定长度的向量逐元素乘以标量值。

- 适用范围：所有架构（无架构特化）
- 使用 `AscendC::SetVectorMask<Element, MaskMode::COUNTER>(len)` 控制有效长度

## 模板原型

```cpp
template <class ArchTag, class VType_>
struct TileVmuls;
```

| 模板参数  | 说明                                              |
| :-------- | :------------------------------------------------ |
| `ArchTag` | 架构标签                                          |
| `VType_`  | 向量数据类型，通过 `VType_::Element` 获取元素类型 |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // 目的 UB LocalTensor
    AscendC::LocalTensor<Element> srcTensor,    // 源 UB LocalTensor
    Element scalar,                             // 标量值
    uint32_t len                                // 向量长度
)
```

## 调用示例

```cpp
#include "catlass/gemv/tile/tile_vmuls.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using VType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t len = 64;
Element scale = 0.5f;

AscendC::LocalTensor<Element> dstTensor, srcTensor;

using VmulsOp = TileVmuls<Arch::AtlasA2, VType>;
VmulsOp vmuls;
vmuls(dstTensor, srcTensor, scale, len);
```
