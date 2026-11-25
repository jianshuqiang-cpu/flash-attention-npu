# Block Epilogue Basic Template
>
> [Code location](../../../../../../../include/catlass/epilogue/block/block_epilogue.hpp)

[TOC]

## BlockEpilogue

### Description

Block-level epilogue. Template parameters such as [`DispatchPolicy`](../../../../../../../include/catlass/gemm/dispatch_policy.hpp) are used to match the corresponding partial specialization branch.

### Template Description

```
template <
    class DispatchPolicy,   // DispatchPolicy used
    class... Args           // Other template parameters
>
struct BlockEpilogue
```
