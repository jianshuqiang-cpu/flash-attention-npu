# CATLASS API List

CATLASS provides a layered GEMM API architecture to assemble templates hierarchically from bottom to top (Basic $\rightarrow$ Tile $\rightarrow$ Block $\rightarrow$ Kernel $\rightarrow$ Device) to implement operators. Developers can reuse low-level components or develop high-level components based on specific requirements to achieve customized operator development.

| Component Classification                                       |                                                                 Description                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------: |
| [gemm/kernel](./include/catlass/gemm/kernel/README.md)         |        Entry point for device-side execution, representing the collective orchestration and execution logic of all blocks on the NPU.        |
| [gemm/block](./include/catlass/gemm/block/README.md)           |                 The primary interface governing the main loop of block-level matrix multiplication and accumulation (MMAD).                  |
| [gemm/tile](./include/catlass/gemm/tile/README.md)             |                             Leverages base APIs to construct the NPU microkernels required for GEMM primitives.                              |
| [epilogue/block](./include/catlass/epilogue/block/README.md)   |                       The block-level epilogue component for GEMM, which can also be applied to non-GEMM computations.                       |
| [epilogue/fusion](./include/catlass/epilogue/fusion/README.md) |                                       The graph orchestrator and foundational node components for EVG.                                       |
| [epilogue/tile](./include/catlass/epilogue/tile/README.md)     |                           Leverages base APIs to construct the NPU microkernels required for epilogue operations.                            |
| [TLA](./include/tla/README.md)                                 | Tensor Layout Architecture. Abstracts underlying data storage details and provides generalized algorithms for multidimensional array access. |
