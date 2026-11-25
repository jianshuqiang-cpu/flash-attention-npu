# VecCopyGmToUB

> [代码位置](../../../../../../../include/catlass/gemv/tile/vec_copy_gm_to_ub.hpp)

[TOC]

## 功能说明

`VecCopyGmToUB` 实现 GEMV（矩阵-向量乘法）场景下从 GM 到 UB 的向量数据搬运。通过 `AscendC::DataCopy` 将长度为 `len` 的一维向量从 Global Memory 拷贝到 Unified Buffer。

- 适用范围：所有架构（无架构特化）
- 单个 block 搬运（`blockCount = 1`），零 stride

## 模板原型

```cpp
template <class ArchTag_, class VType_>
struct VecCopyGmToUB;
```

| 模板参数   | 说明                                              |
| :--------- | :------------------------------------------------ |
| `ArchTag_` | 架构标签                                          |
| `VType_`   | 向量数据类型，通过 `VType_::Element` 获取元素类型 |

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,      // UB 上的目的 LocalTensor
    AscendC::GlobalTensor<Element> srcTensor,     // GM 上的源 GlobalTensor
    uint32_t len                                   // 搬运的元素长度
)
```

## 调用示例

```cpp
#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"

using namespace Catlass::Gemv::Tile;

using Element = half;
using VType = Gemm::GemmType<Element, layout::VectorLayout>;

uint32_t len = 64;

AscendC::GlobalTensor<Element> srcTensor;
AscendC::LocalTensor<Element> dstTensor;

using CopyOp = VecCopyGmToUB<Arch::AtlasA2, VType>;
CopyOp copyOp;
copyOp(dstTensor, srcTensor, len);
```
