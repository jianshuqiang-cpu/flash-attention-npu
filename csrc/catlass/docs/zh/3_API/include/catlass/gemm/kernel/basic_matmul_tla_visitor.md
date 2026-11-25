# Basic Matmul TLA Visitor

> 代码路径：`include/catlass/gemm/kernel/basic_matmul_tla_visitor.hpp`

## 功能说明

这是当前 EVG 在 GM workspace 路径上的 kernel 入口。

执行方式为：

1. AIC 先完成 MMAD，把中间结果写到 GM workspace
2. AIV 等待跨核同步标志
3. AIV 调用 `BlockEpilogue` 执行 EVG

适合普通的 EVG 场景，也是当前仓内多数 EVG 样例采用的入口。

## 模板参数

```cpp
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTlaVisitor;
```

- `BlockMmad_`：GEMM 主循环实现
- `BlockEpilogue_`：EVG 专用尾处理，一般是 `BlockEpilogue<EpilogueVisitor<false>, ...>`
- `BlockScheduler_`：Block 调度器

## Arguments 关键字段

```cpp
struct Arguments {
    GemmCoord problemShape;
    GM_ADDR ptrA; LayoutA layoutA;
    GM_ADDR ptrB; LayoutB layoutB;
    GM_ADDR ptrC; LayoutC layoutC;
    GM_ADDR ptrBias{nullptr};
    typename BlockEpilogue::EVG::Arguments evg_args;
};
```

其中：

- `ptrC/layoutC` 仍保留在公开 `Arguments` 中
- `evg_args` 是 EVG 图的参数

需要注意，当前 visitor kernel 的 `ToUnderlyingArguments()` 实现并不消费 `ptrC/layoutC`，最终写回地址由 `evg_args` 中的 `VisitorAuxStore` 决定。

## Workspace 规则

`GetWorkspaceSize()` 返回：

```cpp
sizeof(ElementC) * M * N + EVG::get_workspace_size(...)
```

前一部分用于保存 MMAD 结果，后一部分用于 EVG 自身节点可能需要的 workspace。

## 适用条件

- `BlockEpilogue_::USE_UB_WORKSPACE` 对应 `false`
- 适用于先把 MMAD 结果落到 GM 再做尾处理的场景
