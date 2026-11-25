# PrologueTraits

> [Code Location](../../../../../../../include/catlass/gemm/tile/tile_traits.hpp)

[TOC]

## Function

`PrologueTraits` is a trait template that unifies the interface of Prologue operators (such as `TileCastInt8ToFp16Dequant`) into a set of Tensor type aliases. This allows upper-level templates like blockMmad to uniformly access the Prologue's input and output Tensor types.

key roles:
- Deduces `TensorSrc`/`TensorDst` as fully-qualified `AscendC::GlobalTensor<T>` types based on `Prologue::ElementSrc`/`Prologue::ElementDst`.
- Provides a partial specialization for `void`, offering a safe, null-type fallback.

## Template Prototype

### Primary Template

```cpp
template <class Prologue>
struct PrologueTraits : public Prologue {
    using Prologue::Prologue;                         // Inherit the constructor.

    using TensorSrc = AscendC::GlobalTensor<typename Prologue::ElementSrc>;
    using TensorDst = AscendC::GlobalTensor<typename Prologue::ElementDst>;
};
```

### void Partial Specialization

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

## Member Types

| Member| Source| Description|
| :------ | :------ | :------ |
| `ElementSrc` | `Prologue::ElementSrc` | Prologue input element type|
| `ElementDst` | `Prologue::ElementDst` | Prologue output element type|
| `LayoutSrc` | `Prologue::LayoutSrc` | Prologue input layout|
| `LayoutDst` | `Prologue::LayoutDst` | Prologue output layout|
| `TensorSrc` | `GlobalTensor<ElementSrc>` | Complete Global Memory (GM) tensor type (input)|
| `TensorDst` | `GlobalTensor<ElementDst>` | Complete GM tensor type (output)|
| `Params` | `Prologue::Params` | Prologue parameter structure|

## Examples

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

// void rollback: compile-time security
using VoidTraits = Tile::PrologueTraits<void>;
// VoidTraits::TensorSrc  → EmptyType
// VoidTraits::TensorDst  → EmptyType
VoidTraits voidTraits; // Variadic constructor, performing no operation
```
