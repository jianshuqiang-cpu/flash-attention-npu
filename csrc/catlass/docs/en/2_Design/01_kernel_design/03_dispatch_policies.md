# Block Dispatch Policies

DispatchPolicy is an important template parameter of BlockMmad. Each DispatchPolicy is defined in `include/catlass/gemm/dispatch_policy`. This document briefly describes the functions, parameters, and examples of the following four DispatchPolicies.

- MmadAtlasA2Pingpong
- MmadAtlasA2Preload
- MmadAtlasA2PreloadAsync
- MmadAtlasA2PreloadAsyncWithCallback

## MmadAtlasA2Pingpong

Function: Sets the ping-pong buffer for L1 and L0A/B in the A2 architecture.

Parameters:

- `STAGES`: number of buffers in a multi-buffer scenario.
- `ENABLE_UNIT_FLAG`: indicates whether to enable `unitflag` optimization and fine-grained parallelism between MMAD computation and the data transfers of L0C results back to GM.

Sample code:

```c++
struct MmadAtlasA2Pingpong {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = true;
};
```

Currently, the examples that use this DispatchPolicy are `00_basic_matmul`, `01_batched_matmul`, `03_matmul_add`, `04_padding_matmul`, and `09_split_matmul`.

## MmadAtlasA2Preload

Function: Uses the ping-pong buffer for L1 and L0A/B in the A2 architecture, and supports the ShuffleK policy and block preloading.

Parameters:

- `STAGES`: number of buffers in a multi-buffer scenario.
- `ENABLE_UNIT_FLAG`: indicates whether to enable `unitflag` optimization and fine-grained parallelism between MMAD computation and the data transfers of L0C results back to GM.
- `ENABLE_SHUFFLE_K`: indicates whether to enable the ShuffleK policy.

Sample code:

```c++
struct MmadAtlasA2Preload {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = true;
    static constexpr bool ENABLE_SHUFFLE_K = true;
};
```

Currently, the example that uses this DispatchPolicy is `06_optimized_matmul`.

## MmadAtlasA2PreloadAsync

Function: Uses the nBuffer for L1 Buffer and L0A/L0B/L0C Buffer in the A2 architecture, and supports the ShuffleK policy, inter-block preloading, and inter-group preloading.

Parameters:

- `PRELOAD_STAGES`: specifies the number of GM-to-L1 data reads before L1-to-L0 data movement and Mmad computation start. The value must be less than that of `L1_STAGES`.
- `L1_STAGES`: specifies the number of buffers enabled for L1. The following condition must be met: L1TileShape's (M\*K\*Number of bytes of the element data type of matrix A + K\*N\*Number of bytes of the element data type of matrix B) <= L1 size
- `L0A_STAGES`: specifies the number of buffers enabled for L0A. The following condition must be met: L0TileShape's (M\*K\*Number of bytes of the element data type of matrix A) <= L0A size
- `L0B_STAGES`: specifies the number of buffers enabled for L0B. The following condition must be met: (K\*N\*Number of bytes of the element data type of matrix B) <= L0B size
- `L0C_STAGES`: specifies the number of buffers enabled for L0C. The following condition must be met: (M\*N\*Number of bytes of the Mmad calculation data type) <= L0C size
- `ENABLE_UNIT_FLAG`: indicates whether to enable `unitflag` optimization and fine-grained parallelism between MMAD computation and the data transfers of L0C results back to GM.
- `ENABLE_SHUFFLE_K`: indicates whether to enable the ShuffleK policy.

Sample code:

```c++
struct MmadAtlasA2PreloadAsync {
    static constexpr uint32_t PRELOAD_STAGES = 1;
    static constexpr uint32_t L1_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = true;
};
```

Currently, the examples that use this DispatchPolicy are `02_grouped_matmul_slice_m`, `05_grouped_matmul_slice_k`, and `11_grouped_matmul_slice_k_per_token_dequant`.

## MmadAtlasA2PreloadAsyncWithCallback

Function: Uses the nBuffer for L1 Buffer and L0A/L0B/L0C Buffer in the A2 architecture, and supports the ShuffleK policy, inter-block preloading, and inter-group preloading. In addition, users can transfer the synchronization commands between AIC and AIV via callbacks to the block layer, and the block layer determines the call timing.

Parameters:

- `PRELOAD_STAGES`: specifies the number of GM-to-L1 data reads before L1-to-L0 data movement and Mmad computation start. The value must be less than that of `L1_STAGES`.
- `L1_STAGES`: specifies the number of buffers enabled for L1. The following condition must be met: (M\*K\*Number of bytes of the element data type of matrix A + K\*N\*Number of bytes of the element data type of matrix B) <= L1 size
- `L0A_STAGES`: specifies the number of buffers enabled for L0A. The following condition must be met: (M\*K\*Number of bytes of the element data type of matrix A) <= L0A size
- `L0B_STAGES`: specifies the number of buffers enabled for L0B. The following condition must be met: (K\*N\*Number of bytes of the element data type of matrix B) <= L0B size
- `L0C_STAGES`: specifies the number of buffers enabled for L0C. The following condition must be met: (M\*N\*Number of bytes of the Mmad calculation data type) <= L0C size
- `ENABLE_UNIT_FLAG`: indicates whether to enable `unitflag` optimization and fine-grained parallelism between MMAD computation and the data transfers of L0C results back to GM.
- `ENABLE_SHUFFLE_K`: indicates whether to enable the ShuffleK policy.

Sample code:

```c++
struct MmadAtlasA2PreloadAsyncWithCallback {
    static constexpr uint32_t PRELOAD_STAGES = 1;
    static constexpr uint32_t L1_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = true;
};
```

Currently, the examples that use this DispatchPolicy are `10_grouped_matmul_slice_m_per_token_dequant` and `12_quant_matmul`.
