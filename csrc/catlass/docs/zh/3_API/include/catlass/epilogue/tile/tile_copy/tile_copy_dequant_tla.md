# TileCopyDequantTla

> [代码位置](../../../../../../../../include/catlass/epilogue/tile/tile_copy.hpp)

[TOC]

## 功能说明

`TileCopyDequantTla` 是 epilogue TLA 风格反量化的搬运聚合模板。引用 `CopyGm2UbTla` 和 `CopyUb2GmTla` 作为子组件，每个子组件为需要 `TensorSrc`/`TensorDst` 类型参数化延迟实例化的模板。

- 适用范围：AtlasA2、Ascend950
- 风格：TLA

## 模板原型

```cpp
template <
    class ArchTag,
    class ElementC_,   class LayoutTagC_,
    class ElementX_,   class LayoutTagX_,
    class ElementY_,   class LayoutTagY_,
    class ElementD_,   class LayoutTagD_
>
struct TileCopyDequantTla;
```

## 成员类型定义

| 成员类型              | 说明                                     |
| :-------------------- | :--------------------------------------- |
| `CopyGmToUbC`（模板） | `CopyGm2UbTla<Arch, TensorC, TensorUbC>` |
| `CopyGmToUbX`（模板） | `CopyGm2UbTla<Arch, TensorX, TensorUbX>` |
| `CopyGmToUbY`（模板） | `CopyGm2UbTla<Arch, TensorY, TensorUbY>` |
| `CopyUbToGmD`（模板） | `CopyUb2GmTla<Arch, TensorUbD, TensorD>` |
