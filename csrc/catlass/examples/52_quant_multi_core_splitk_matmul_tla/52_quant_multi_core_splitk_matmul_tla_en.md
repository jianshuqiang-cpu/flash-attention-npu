# CATLASS Quant Multi Core SplitK Matmul TLA Example Introduction

## Prototype design

| Name          | Type      | Data Type | Dimension | Format | Description                           |
| ------------- | --------- | --------- | --------- | ------ | ------------------------------------- |
| matA          | inTensor  | int8      | [m, k]    | ND     | Left matrix                           |
| matB          | inTensor  | int8      | [n, k]    | ND     | Right matrix. Transpose is supported. |
| scale         | inTensor  | float     | [n]       | ND     | Per-channel quantization scale        |
| perTokenScale | inTensor  | float     | [m]       | ND     | Per-token quantization scale          |
| matD          | outTensor | bf16      | [m, n]    | ND     | Output matrix                         |

## Sample Implementation

The CATLASS [`52_quant_multi_core_splitk_matmul_tla` example](./README_en.md) operator is an Ascend-friendly Matmul operator implemented based on the CATLASS Gemm API. It is optimized for large matrix computation scenarios. The key operator components include the following parts:

- Example assembly: [quant_multi_core_splitk_matmul_tla.cpp](./quant_multi_core_splitk_matmul_tla.cpp)
- Kernel implementation:
  - Main kernel file: [quant_multi_core_splitk_matmul_tla.hpp](../../include/catlass/gemm/kernel/quant_multi_core_splitk_matmul_tla.hpp)

- Block components, including:
  - Basic mmad component [block_mmad_pingpong_tla.hpp (replaceable)](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)
  - Dequantization post-processing component [block_epilogue_per_token_dequant_tla.hpp](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant_tla.hpp)

## Solution Design

For the basic multi-core K-splitting solution, refer to [this guide](../102_dynamic_optimized_matmul/docs/en/MultiCoreSplitkMatmul_en.md). This section mainly describes two solutions for adapting quantization post-processing in the multi-core K-splitting scenario. This example is implemented according to `Plan B`.

![Multi-core-to-K fusion quantization post-processing](https://raw.gitcode.com/user-images/assets/7801479/3116a627-a26f-422a-a11f-e1b4eb647bd7/quant_splitk.png "quant_splitk.png")

The computation on the AIV core is divided into two parts: 1. After the multi-core K-splitting matrix multiplication, there is a ReduceAdd accumulation along the K axis; 2. Dequantization of the matrix multiplication result `In Plan A`, AIC performs matrix multiplication, AIV performs dequantization, and then performs accumulation. `In Plan B`, AIC performs matrix multiplication, AIV performs accumulation, and then performs dequantization. For scenarios where each core has multiple rounds of basic block computations, `Plan A` can benefit from overlapping computation with CV. If each core only performs one round of basic block computation, `Plan B` requires less dequantization computation.

## Performancem Benefits

Compared with the 12_quant_matmul operator, the optimization lies in achieving load balancing through K-splitting when the C matrix is small. The disadvantage is that dequantization is handled separately at the end, without CV fusion hiding, and adds the ReduceAdd operation on the AI core for accumulation along the K direction.

| M   | N    | K     | Baseline Time (μs) | Time (μs) | Speedup ratio |
| --- | ---- | ----- | ------------------ | --------- | ------------- |
| 128 | 256  | 1024  | 18.64              | 18.11     | 1.03          |
| 128 | 256  | 2048  | 23.23              | 20.67     | 1.12          |
| 128 | 256  | 4096  | 32.97              | 21.15     | 1.56          |
| 128 | 256  | 8192  | 47.73              | 20.01     | 2.39          |
| 128 | 256  | 16384 | 78.52              | 23.83     | 3.30          |
| 256 | 512  | 16384 | 75.39              | 34.29     | 2.20          |
| 256 | 1024 | 16384 | 77.35              | 51.41     | 1.50          |

- The baseline is the QuantMatmul operator.
- The times reported are total kernel function times, obtained using the msprof tool.
- In the preceding test case, matrix A is in `layout::RowMajor` format, and matrix B is in `layout::ColumnMajor` format.
- Test environment description: The NPU model is 910B3, and the CANN package version is 8.5.0.
