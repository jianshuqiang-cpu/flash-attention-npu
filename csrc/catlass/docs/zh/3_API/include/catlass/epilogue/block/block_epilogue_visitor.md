# Block Epilogue Visitor 偏特化

> 代码位置：`include/catlass/epilogue/block/block_epilogue_visitor.hpp`

## 功能说明

这是 EVG 使用的 `BlockEpilogue` 偏特化实现。

它负责把一个 block 的结果继续切成更小的 tile，并按 `LOAD -> COMPUTE -> STORE` 的顺序驱动 EVG 在每个 tile 上执行，同时通过两套 callbacks 组织双缓冲流水。

## 对应模板形态

EVG 使用的 `BlockEpilogue` 形态如下：

```cpp
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueVisitor<false>,
    ArchTag,
    Int<computeLength>,
    EVG,
    ElementC
>;
```

或：

```cpp
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueVisitor<true>,
    ArchTag,
    Int<computeLength>,
    EVG,
    ElementC
>;
```

对应的偏特化模板为：

```cpp
template <
    bool USE_UB_WORKSPACE_,
    class ArchTag_,
    class ComputeLength_,
    class EVG_,
    class ElementC_
>
class BlockEpilogue<
    EpilogueVisitor<USE_UB_WORKSPACE_>,
    ArchTag_,
    ComputeLength_,
    EVG_,
    ElementC_
>;
```

## 关键职责

- 接收 `EVG::Params`
- 把 block 结果切成 tile
- 为 EVG 分配两套 callbacks
- 以 `LOAD -> COMPUTE -> STORE` 顺序驱动每个 tile
- 用事件同步实现双缓冲

## 关键参数

- `EpilogueVisitor<false>`：从 GM workspace 读取 MMAD 结果
- `EpilogueVisitor<true>`：直接从 UB 读取 MMAD 结果
- `ComputeLength`：单次 tile 处理元素数，要求满足对齐约束
- `EVG`：完整的尾处理图
- `ElementC`：MMAD 输出类型

## 执行要点

- 当 `USE_UB_WORKSPACE == false` 时，EVG 从 GM workspace 读取 `C`
- 当 `USE_UB_WORKSPACE == true` 时，EVG 直接使用 UB 中的 MMAD 结果
- 内部会按当前 tile 宽度自动选择“多行整列处理”或“逐行分列处理”
- 两套 callbacks 会交替工作，用来形成 tile 级的双缓冲流水

## 相关文档

- [block_epilogue](./block_epilogue.md)
- [fusion/README](../fusion/README.md)
- [evg_api](../../../../evg_api.md)
