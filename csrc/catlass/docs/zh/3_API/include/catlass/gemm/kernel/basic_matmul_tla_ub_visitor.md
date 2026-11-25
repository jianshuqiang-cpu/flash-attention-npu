# Basic Matmul TLA UB Visitor

> 代码路径：`include/catlass/gemm/kernel/basic_matmul_tla_ub_visitor.hpp`

## 功能说明

这是当前 EVG 在 UB workspace 路径上的 kernel 入口。

执行方式为：

1. AIC 在 UB 中完成 MMAD
2. 通过跨核同步把时序交给 AIV
3. AIV 直接消费 UB 中的结果并执行 EVG

相比 GM workspace 路径，它省掉了把 MMAD 结果整体写回 GM 再读出的过程。

## 模板参数

```cpp
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTlaUbVisitor;
```

常见搭配：

- `BlockEpilogue_` 使用 `EpilogueVisitor<true>`
- `VisitorAccLoad` 也打开 UB workspace 模式

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

同样需要注意，当前 visitor kernel 的 `ToUnderlyingArguments()` 实现并不消费 `ptrC/layoutC`，最终输出位置由 `evg_args` 中的 `VisitorAuxStore` 决定。

## Workspace 规则

`GetWorkspaceSize()` 只返回：

```cpp
EVG::get_workspace_size(...)
```

因为 MMAD 结果不再额外落到 GM workspace。

## 适用条件

- 当前实现限制为 `Arch::Ascend950`
- `BlockEpilogue::USE_UB_WORKSPACE` 对应 `true`
- 更适合希望减少中间结果回写开销的场景
