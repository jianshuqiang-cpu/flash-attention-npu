# BasicConv2dTla Example Readme

## 代码组织

```text
├── 56_ascend950_basic_conv2d_tla
│   ├── CMakeLists.txt   # CMake编译文件
│   ├── README.md
│   └── basic_conv2d_tla.cpp # 主文件
```

## 功能说明

- 算子功能：完成基础2D版本卷积计算（使用[`TLA`](../../docs/zh/2_Design/02_tla/01_layout.md)语义）
- 该用例总体设计与[`33_basic_conv2d`](../33_basic_conv2d/README.md)相同，区别为使用了TLA相关抽象，条件约束与[`BasicConv2d` 约束](../33_basic_conv2d/README.md#功能说明)一致，注意在Ascend950硬件条件下，`L0C_SIZE`为256K。


## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#算子编译)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

执行结果如下，表明精度验证通过。

```text
Compare success.
```
