# MXFP8MatmulTla Example Readme

## 功能介绍

- 演示 Ascend 950 上的 **MX FP8 矩阵乘**：A、B 为 MX FP8，经 `float8_e8m0` 缩放后做矩阵乘，输出 FP32。
- 本示例中 A、B 元素类型为 `float8_e4m3_t`；缩放因子为 `float8_e8m0_t`。未启用 Bias（`ElementBias` 为 `void`）。
- 默认布局为 A `RowMajor`、B `ColumnMajor`、C `RowMajor`，与 `gen_data.py` 在 `trans_a=0, trans_b=1` 时生成的数据一致。

## 代码组织

```text
├── 53_ascend950_fp8_mx_matmul
│   ├── CMakeLists.txt      # CMake 编译配置
│   ├── README.md
│   ├── gen_data.py         # 生成 input/ 与 golden/
│   ├── fp8_mx_matmul.cpp   # 主程序
│   └── fp8_mx_matmul_aswt.cpp   # BlockSchedulerAswt 变体
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 53_ascend950_fp8_mx_matmul -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/53_ascend950_fp8_mx_matmul/data 下生成 input/ 与 golden/）
python3 examples/53_ascend950_fp8_mx_matmul/gen_data.py 256 512 1024 0 1
# 可选：--data-root <DIR> 指定在 DIR/data/ 下生成（默认在脚本所在目录下生成）
# 输入参数分别对应 m, n, k, trans_a, trans_b
# trans_a表示A矩阵是否转置，0是不转置，1是转置
# trans_b表示B矩阵是否转置，0是不转置，1是转置
# 执行测试样例
./output/bin/53_ascend950_fp8_mx_matmul 256 512 1024 0
# ASWT 调度变体（与上共用同一 data/，需先 gen_data）
bash scripts/build.sh 53_ascend950_fp8_mx_matmul_aswt -DCATLASS_ARCH=3510
./output/bin/53_ascend950_fp8_mx_matmul_aswt 256 512 1024 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
```

执行结果如下，说明精度比对成功。

```cpp
Compare success.
```

## 使用说明

1、 `gen_data.py`的输入支持trans_a和trans_b，但53_ascend950_fp8_mx_matmul可执行文件不支持，仅仅是trans_a为0及trans_b为1的example示例。

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
A、B支持数据类型为float8_e4m3或float8_e5m2
MxScaleA、MxScaleB支持数据类型为float8_e8m0

其中对于MxScaleA、MxScaleB的数据排布要求如下：
当A为RowMajor时，MxScaleA的shape为（m, ceil(k/64), 2）
当A为ColumnMajor时，MxScaleA的shape为（ceil(k/64), m, 2）
当B为RowMajor时，MxScaleB的shape为（ceil(k/64), n, 2）
当B为ColumnMajor时，MxScaleB的shape为（n, ceil(k/64), 2）

3、 MxMatmul默认使用的DispatchPolicy MxMmad支持以下几个模板参数：

| 模板参数         | 默认值 | 参数说明                                         |
| ---------------- | ------ | ------------------------------------------------ |
| ArchTag          | 无     | 指定架构型号                                     |
| enableUnitFlag   | false  | 是否开启Unitflag，开启L0C多缓冲时必须设置为false |
| l0CStages        | 1      | 指定L0C的缓冲区数量，设置为2即可开启L0C双缓冲    |
| enableL1Resident | false  | 是否开启L1常驻                                   |
| l1AStages        | 2      | L1上加载矩阵A的Buffer数量                        |
| l1BStages        | 2      | L1上加载矩阵B的Buffer数量                        |
| l0AStages        | 2      | L0上加载矩阵A的Buffer数量                        |
| l0BStages        | 2      | L0上加载矩阵B的Buffer数量                        |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTiles = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTiles = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
