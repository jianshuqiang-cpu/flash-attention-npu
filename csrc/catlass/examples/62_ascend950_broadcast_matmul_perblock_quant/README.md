# BroadcastMatmulPerblockQuant Example Readme

## 代码组织

```text
├── 62_ascend950_broadcast_matmul_perblock_quant
│   ├── CMakeLists.txt              # CMake编译文件
│   ├── README.md
│   ├── gen_data_compare.py         # 数据生成+精度比对脚本
│   └── broadcast_matmul_perblock_quant_tla.cpp # 算子调用示例
```

## 功能说明

该算子实现了张量A (shape [B,M,K])和矩阵B(shape [K,N])的广播矩阵乘法，并对计算结果进行perblock量化(block大小为[M,K])。
算子典型应用场景为Q,K,V与旋转矩阵进行矩阵乘法以平滑数据分布，然后进行MXFP8(E4M3)量化。

## 参数说明

| 参数名 | 输入/输出 | 描述                             | 数据类型      | 数据格式 | Layout       |
| ------ | --------- | -------------------------------- | ------------- | -------- | ------------ |
| a      | 输入      | 张量a                            | bfloat16      | ND       | RowMajor     |
| b      | 输入      | 矩阵b                            | bfloat16      | ND       | RowMajor     |
| out    | 输出      | a与b的广播矩阵乘法的量化结果     | float8_e4m3fn | ND       | RowMajor     |
| scale  | 输出      | a与b的广播矩阵乘法的量化缩放系数 | float32       | ND       | VectorLayout |

- 输入a的shape为[B,M,K]
- 输入b的shape为[K,N]
- 输出out的shape为[B,M,N]
- 输出scale的shape为[B]

## 约束说明

- B的取值范围为[1,65536]; 对应Q,K,V按照MXFP8量化分块后block的数量。
- M的取值范围为{128,256}; 对应MXFP8量化的block大小。
- N和K的取值范围为{128}; 对应旋转矩阵的大小

## 使用示例

### 数据生成与精度比对

```bash
# 编译指定用例
bash scripts/build.sh 62_ascend950_broadcast_matmul_perblock_quant -DCATLASS_ARCH=3510

# 在示例目录下运行数据生成和比对脚本
cd examples/62_ascend950_broadcast_matmul_perblock_quant

# python3 gen_data_compare.py <batch_count> <m> <n> <k> <device_id>
python3 gen_data_compare.py 1024 128 128 128 0
```

执行结果如下，说明dst和scale精度比对成功。

```text
------ 生成测试数据 ------
batch_count=1024, m=128, n=128, k=128
------ 运行NPU算子 ------
npu op run log =
------ 比对结果 ------
------ 计算相对误差 -----
------ 综合精度指标 ------
dst: npu mare=0.1250, golden mare=0.125000
dst: npu mere=0.0018, golden mere=0.001839
dst: npu rmse=1.3765, golden rmse=1.377344
scale: npu mare=0.0030, golden mare=0.003028
scale: npu mere=0.0013, golden mere=0.001291
scale: npu rmse=0.0006, golden rmse=0.000632
------ 开始比较 ------
精度指标比较结果：Compare success
```
