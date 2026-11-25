# TileMuls

> [代码位置](../../../../../../../include/catlass/gemm/tile/tile_muls.hpp)

[TOC]

## 功能说明

`TileMuls` 模板完成了 Vector 引擎的标量乘法（`AscendC::Muls`），对 UB 上某个 tensor 的全部元素乘以一个标量 `scalar`，结果写回 UB。

适用的场景：在 Epilogue 阶段对 L0C→UB 后累积结果进行 scale 缩放。

该模板使用 `AscendC::SetVectorMask` + `AscendC::Muls` 的组合，通过 COUNTER 掩码模式精确控制计算长度，避免越界访问。

## 模板原型

```cpp
template <
    class ArchTag_,              // 架构标签
    class ComputeType_,          // 计算类型：Gemm::GemmType<Element, Layout>
    uint32_t COMPUTE_LENGTH_     // 单次计算长度
>
struct TileMuls {
    using Element = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
};
```

> `COMPUTE_LENGTH_` 为模板参数传入的常量，不通过运行时参数控制，用于常量折叠优化。

## 调用接口

```cpp
void operator()(
    AscendC::LocalTensor<Element> dstTensor,    // UB 目标 tensor
    AscendC::LocalTensor<Element> srcTensor,    // UB 源 tensor
    Element scalar,                              // 标量
    uint32_t len                                 // 实际计算长度
);
```

执行流程：

1. `SetMaskCount()` → `SetVectorMask<Element, COUNTER>(len)` 设置掩码
2. `Muls<Element, false>(dst, src, scalar, MASK_PLACEHOLDER, 1, {} )`
3. `SetMaskNorm()` → `ResetMask()` 恢复掩码

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_muls.hpp"

using namespace Catlass::Gemm;

using ComputeType = Gemm::GemmType<half, layout::RowMajor>;
constexpr uint32_t COMPUTE_LENGTH = 256;

using MulsOp = Tile::TileMuls<Arch::AtlasA2, ComputeType, COMPUTE_LENGTH>;

half scalar = 0.5_hf;
uint32_t len = 256;

AscendC::LocalTensor<half> srcUB;
AscendC::LocalTensor<half> dstUB;

MulsOp mulsOp;
mulsOp(dstUB, srcUB, scalar, len);

// 等效于：dstUB[i] = srcUB[i] * 0.5  for i in [0, len)
```
