# QuantMatmulFullLoadA Example Readme

## 代码组织

```text
├── 52_quant_multi_core_splitk_matmul_tla
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── 52_quant_multi_core_splitk_matmul_tla.md # 设计文档
│   └── quant_multi_core_splitk_matmul_tla.cpp # 主文件
```

## 功能介绍

该模板为量化多核切K模板，通过切分K，划分出更多的任务块，从而利用更多的计算核心，并使用了TLA相关抽象，因此提供相关示例说明。

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 52_quant_multi_core_splitk_matmul_tla
cd output/bin
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
./52_quant_multi_core_splitk_matmul_tla 256 512 1024 0
```

执行结果如下，说明精度比对成功。

```text
Compare success.
```
