# Basic Matmul TLA UB Visitor
>
> Code path: `include/catlass/gemm/kernel/basic_matmul_tla_ub_visitor.hpp`

## Description

This is the kernel entry point for EVG utilizing the UB workspace path.

The execution workflow is as follows:

1. AIC completes the MMAD computation in the UB.
2. The execution sequence is handed over to the AIV through cross-core synchronization.
3. AIV directly consumes the results from the UB and executes the EVG processing.

Compared to the GM workspace path, this pipelining eliminates the overhead of writing the entire intermediate MMAD result back to the GM and reading it back out.

## Template Parameters

```cpp
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTlaUbVisitor;
```

Common configurations:

- `BlockEpilogue_` is configured to use `EpilogueVisitor<true>`.
- `VisitorAccLoad` also has its UB workspace mode enabled.

## Key Fields of Arguments

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

Note that the `ToUnderlyingArguments()` implementation of the current visitor kernel does not consume `ptrC` or `layoutC`. The final destination writeback address is determined by the `VisitorAuxStore` contained within `evg_args`.

## Workspace Rules

GetWorkspaceSize() returns only:

```cpp
EVG::get_workspace_size(...)
```

This is because intermediate MMAD results no longer require extra storage allocation in the GM workspace.

## Usage Conditions

- The current implementation is constrained to Arch::Ascend950.
- `BlockEpilogue::USE_UB_WORKSPACE` must evaluate to `true`.
- This path is suitable for scenarios where minimizing intermediate writeback latency is a priority.
