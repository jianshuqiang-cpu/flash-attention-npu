# A8W4MxMatmul Example Readme

## 功能介绍

- 演示 Ascend 950 上的伪量化场景下的Mx Matmul矩阵乘法：左矩阵 A与伪量化后的右矩阵 B 经 MX 缩放（`float8_e8m0`）后在 Cube 上完成乘加，输出为 FP32。
- 本示例中 A元素类型为`float8_e4m3_t`，B元素类型为 `float4_e2m1x2_t`；缩放因子为 `float8_e8m0_t`。未启用 Bias（`ElementBias` 为 `void`）。
- 默认布局为 A `RowMajor`、B `ColumnMajor`、C `RowMajor`，与 `gen_data.py` 在 `trans_a=0, trans_b=1` 时生成的数据一致。

## 代码组织

```text
├── 59_ascend950_a8w4_mx_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── gen_data.py
│   └── a8w4_mx_matmul.cpp # 主文件
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 59_ascend950_a8w4_mx_matmul -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/59_ascend950_a8w4_mx_matmul/data 下生成 input/ 与 golden/）
python3 examples/59_ascend950_a8w4_mx_matmul/gen_data.py 128 128 128 0 1
# 输入参数分别对应 m, n, k, trans_a, trans_b
# trans_a表示A矩阵是否转置，0是不转置，1是转置
# trans_b表示B矩阵是否转置，0是不转置，1是转置
# 执行测试样例
./output/bin/59_ascend950_a8w4_mx_matmul 128 128 128 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
```

执行结果如下，说明精度比对成功。

```cpp
Compare success.
```

## 使用说明

1、 `gen_data.py`的输入支持trans_a和trans_b，但59_ascend950_a8w4_mx_matmul可执行文件不支持，仅仅是trans_a为0及trans_b为1的example示例。

若要对应转置情况请修改example示例中的layout，因为layout隐式表征转置状态，即layout::RowMajor表示不转置，layout::ColumnMajor表示转置。

其对应关系如下表：

| trans_a | trans_b | LayoutA             | LayoutB             |
| ------- | ------- | ------------------- | ------------------- |
| 0       | 0       | layout::RowMajor    | layout::RowMajor    |
| 0       | 1       | layout::RowMajor    | layout::ColumnMajor |
| 1       | 0       | layout::ColumnMajor | layout::RowMajor    |
| 1       | 1       | layout::ColumnMajor | layout::ColumnMajor |

2、 本example完成mx量化矩阵乘：
C = (MxScaleA x A) * (MxScaleB x B) + Bias
A、B支持数据类型为float8_e4m3和float4_e2m1，B矩阵伪量化为float8_e4m3后参与cube计算
MxScaleA、MxScaleB支持数据类型为float8_e8m0

其中对于MxScaleA、MxScaleB的数据排布要求如下：
当A为RowMajor时，MxScaleA的shape为（m, ceil(k/64), 2）
当A为ColumnMajor时，MxScaleA的shape为（ceil(k/64), m, 2）
当B为RowMajor时，MxScaleB的shape为（ceil(k/64), n, 2）
当B为ColumnMajor时，MxScaleB的shape为（n, ceil(k/64), 2）
