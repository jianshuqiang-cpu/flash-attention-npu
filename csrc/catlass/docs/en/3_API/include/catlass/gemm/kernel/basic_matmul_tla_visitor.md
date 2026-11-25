# Basic Matmul TLA Visitor
>
> Code path: `include/catlass/gemm/kernel/basic_matmul_tla_visitor.hpp`

## Description

This is the kernel entry point for EVG utilizing the GM workspace path.

The execution workflow is as follows:

1. AI Core completes the MMAD computation and writes the intermediate results to the GM workspace.
2. AIV waits for the cross-core synchronization flag.
3. AIV invokes `BlockEpilogue` to execute the EVG processing.

This entry point is designed for standard EVG scenarios and is used by most EVG examples in the current repository.

## Template Parameters

```cpp
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class BasicMatmulTlaVisitor;
```

- `BlockMmad_`: GEMM main loop implementation
- `BlockEpilogue_`: EVG-specific epilogue, typically configured as `BlockEpilogue<EpilogueVisitor<false>, ...>`
- `BlockScheduler_`: Block scheduler

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

Where:

- `ptrC` and `layoutC` are still reserved in the public `Arguments` structure.
- `evg_args` contains the specific execution parameters for the EVG graph.

Note that the `ToUnderlyingArguments()` implementation of the current visitor kernel does not consume `ptrC` or `layoutC`. The actual writeback logic and destination address are governed by the `VisitorAuxStore` contained within `evg_args`.

## Workspace Rules

`GetWorkspaceSize()` returns:

```cpp
sizeof(ElementC) * M * N + EVG::get_workspace_size(...)
```

The first term allocates space to store the MMAD results, while the second term accounts for the workspace required by individual EVG nodes.

## Usage Conditions

- `BlockEpilogue_::USE_UB_WORKSPACE` must evaluate to `false`.
- This is applicable to scenarios where MMAD results are written out to the GM before epilogue is executed.
