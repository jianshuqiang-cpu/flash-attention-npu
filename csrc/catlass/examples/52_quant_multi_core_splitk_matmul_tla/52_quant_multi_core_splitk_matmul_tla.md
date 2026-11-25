# CATLASS Quant Multi Core SplitK Matmul Tla 样例介绍

## 原型设计

| 名称/Name     | 类型/Class | 数据类型/Dtype | 维度/Dims | 格式/Format | 描述/Description   |
| ------------- | ---------- | -------------- | --------- | ----------- | ------------------ |
| matA          | inTensor   | int8           | [m, k]    | ND          | 左矩阵             |
| matB          | inTensor   | int8           | [n, k]    | ND          | 右矩阵，支持转置   |
| scale         | inTensor   | float          | [n]       | ND          | perChannel量化系数 |
| perTokenScale | inTensor   | float          | [m]       | ND          | perToken量化系数   |
| matD          | outTensor  | bf16           | [m, n]    | ND          | 输出矩阵           |

## 样例实现

CATLASS [`52_quant_multi_core_splitk_matmul_tla`样例](./README.md)算子是基于CATLASS Gemm API实现的昇腾亲和Matmul算子，针对大尺寸矩阵计算场景优化设计，关键算子组件包括以下几部分:

- **Example组装**：[quant_multi_core_splitk_matmul_tla.cpp](./quant_multi_core_splitk_matmul_tla.cpp)
- **Kernel实现**：
  - 主Kernel文件：[quant_multi_core_splitk_matmul_tla.hpp](../../include/catlass/gemm/kernel/quant_multi_core_splitk_matmul_tla.hpp)

- **Block组件**，包含：
  - 基础的mmad组件[block_mmad_pingpong_tla.hpp（可替换）](../../include/catlass/gemm/block/block_mmad_pingpong_tla.hpp)
  - 反量化后处理组件[block_epilogue_per_token_dequant_tla.hpp](../../include/catlass/epilogue/block/block_epilogue_per_token_dequant_tla.hpp)

## 方案设计

基础的多核切K方案[参考](../102_dynamic_optimized_matmul/docs/zh/MultiCoreSplitkMatmul.md)，这里主要说明多核切K下量化后处理适配的两种方案，且本样例根据`Plan B`实现。

![多核切K融合quant后处理示意图](https://raw.gitcode.com/user-images/assets/7801479/3116a627-a26f-422a-a11f-e1b4eb647bd7/quant_splitk.png "quant_splitk.png")

算子在AIV上的计算为两块：①矩阵乘多核切K后，沿K轴的ReduceAdd累加；②矩阵乘结果的反量化。`Plan A`是AIC完成矩阵乘后、AIV先做反量化再做累加，`Plan B`是AIC完成矩阵乘后、AIV先做累加再做反量化。对于每个核存在多轮基本块计算的场景，`Plan A`可以有CV互相掩盖的收益；若每个核仅一轮基本块计算，`Plan B`的反量化计算量更少。

## 性能收益

性能和 12_quant_matmul 比较，优化点在于C矩阵较小时通过切K实现负载均衡，劣化点在于将反量化放在最后单独处理，没有cv融合掩盖，并增加了AIV的ReduceAdd动作完成K方向上累加

| M   | N    | K     | 标杆耗时(us) | 耗时(us) | 加速比 |
| --- | ---- | ----- | ------------ | -------- | ------ |
| 128 | 256  | 1024  | 18.64        | 18.11    | 1.03   |
| 128 | 256  | 2048  | 23.23        | 20.67    | 1.12   |
| 128 | 256  | 4096  | 32.97        | 21.15    | 1.56   |
| 128 | 256  | 8192  | 47.73        | 20.01    | 2.39   |
| 128 | 256  | 16384 | 78.52        | 23.83    | 3.30   |
| 256 | 512  | 16384 | 75.39        | 34.29    | 2.20   |
| 256 | 1024 | 16384 | 77.35        | 51.41    | 1.50   |

- 标杆为QuantMatmul算子。
- 统计耗时均为核函数总耗时，使用msprof工具得到。
- 上述测试例中A矩阵为`layout::RowMajor`排布方式，B矩阵为`layout::ColumnMajor`排布方式。
- 测试环境说明：NPU型号为910B3，CANN包版本为8.5.0。
