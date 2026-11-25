# PrologueTraits

> [代码位置](../../../../../../../include/catlass/gemm/tile/tile_traits.hpp)

[TOC]

## 功能说明

`PrologueTraits` 是一个 trait 模板，将 Prologue 算子（如 `TileCastInt8ToFp16Dequant`）的接口包装为统一的 Tensor 类型别名，方便上层模板（如 blockMmad）获取 Prologue 相关的输入/输出 Tensor 类型。

关键作用：

- 从 `Prologue::ElementSrc` / `Prologue::ElementDst` 推导出 `TensorSrc` / `TensorDst` 的 `AscendC::GlobalTensor<T>` 完整类型
- 对 `void` 偏特化，提供安全的空类型回退

## 模板原型

### 主模板

```cpp
template <class Prologue>
struct PrologueTraits : public Prologue {
    using Prologue::Prologue;                         // 继承构造函数

    using TensorSrc = AscendC::GlobalTensor<typename Prologue::ElementSrc>;
    using TensorDst = AscendC::GlobalTensor<typename Prologue::ElementDst>;
};
```

### void 偏特化

```cpp
template <>
struct PrologueTraits<void> {
    using ElementSrc = EmptyType;
    using LayoutTagSrc = EmptyType;
    using ElementDst = EmptyType;
    using LayoutTagDst = EmptyType;

    using TensorSrc = EmptyType;
    using TensorDst = EmptyType;

    using Params = EmptyType;

    template <class... Args>
    CATLASS_DEVICE
    PrologueTraits(Args...) {}
};
```

## 成员类型

| 成员         | 来源                       | 说明                  |
| :----------- | :------------------------- | :-------------------- |
| `ElementSrc` | `Prologue::ElementSrc`     | Prologue 输入元素类型 |
| `ElementDst` | `Prologue::ElementDst`     | Prologue 输出元素类型 |
| `LayoutSrc`  | `Prologue::LayoutSrc`      | Prologue 输入 layout  |
| `LayoutDst`  | `Prologue::LayoutDst`      | Prologue 输出 layout  |
| `TensorSrc`  | `GlobalTensor<ElementSrc>` | 完整 GM Tensor 类型   |
| `TensorDst`  | `GlobalTensor<ElementDst>` | 完整 GM Tensor 类型   |
| `Params`     | `Prologue::Params`         | Prologue 参数结构体   |

## 调用示例

```cpp
#include "catlass/gemm/tile/tile_traits.hpp"
#include "catlass/gemm/tile/cast_int8_to_fp16.hpp"

using namespace Catlass::Gemm;

using PrologueType = Tile::TileCastInt8ToFp16Dequant<
    Arch::AtlasA2,
    Gemm::GemmType<int8_t, layout::RowMajor>,
    Gemm::GemmType<half, layout::RowMajor>,
    1024>;

using Traits = Tile::PrologueTraits<PrologueType>;

// Traits::ElementSrc  → int8_t
// Traits::ElementDst  → half
// Traits::TensorSrc   → AscendC::GlobalTensor<int8_t>
// Traits::TensorDst   → AscendC::GlobalTensor<half>
// Traits::Params      → TileCastInt8ToFp16Dequant::Params

// void 回退：编译期安全
using VoidTraits = Tile::PrologueTraits<void>;
// VoidTraits::TensorSrc  → EmptyType
// VoidTraits::TensorDst  → EmptyType
VoidTraits voidTraits;  // 变参构造函数，无操作
```
