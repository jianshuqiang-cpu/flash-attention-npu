# MXFP4MatmulTla Example Readme

## 功能介绍

- 演示 Ascend 950 上的 **MX FP4 矩阵乘**：左矩阵 A、右矩阵 B 经 MX 缩放（`float8_e8m0`）后在 Cube 上完成乘加，输出为 FP32。
- 本示例中 A、B 元素类型为 `float4_e2m1x2_t`（E2M1 打包）；缩放因子为 `float8_e8m0_t`。未启用 Bias（`ElementBias` 为 `void`）。
- 默认布局为 A `RowMajor`、B `ColumnMajor`、C `RowMajor`，与 `gen_data.py` 在 `trans_a=0, trans_b=1` 时生成的数据一致。

## 代码组织

```text
├── 54_ascend950_fp4_mx_matmul
│   ├── CMakeLists.txt     # CMake编译文件
│   ├── README.md
│   ├── gen_data.py
│   ├── fp4_mx_matmul.cpp # 主文件
│   └── fp4_mx_matmul_aswt.cpp   # BlockSchedulerAswt 变体
```

## 使用示例

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)，本用例为 Ascend950（3510）算子，编译时需加 `-DCATLASS_ARCH=3510`。L1 分块为 256×256×448、L0 为 256×256×128，以满足 512KiB L1 与 L0 容量约束（勿随意增大 L1 的 K，否则 `L1TileShape exceeding the L1 space`）。
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 54_ascend950_fp4_mx_matmul -DCATLASS_ARCH=3510
# 生成测试样例（在 examples/54_ascend950_fp4_mx_matmul/data 下生成 input/ 与 golden/）
python3 examples/54_ascend950_fp4_mx_matmul/gen_data.py 256 512 1024 0 1
# 可选：--data-root <DIR> 指定在 DIR/data/ 下生成（默认在脚本所在目录下生成）
# 输入参数分别对应 m, n, k, trans_a, trans_b
# trans_a表示A矩阵是否转置，0是不转置，1是转置
# trans_b表示B矩阵是否转置，0是不转置，1是转置
# 执行测试样例
./output/bin/54_ascend950_fp4_mx_matmul 256 512 1024 0
# ASWT 调度变体（与上共用同一 data/，需先 gen_data）
bash scripts/build.sh 54_ascend950_fp4_mx_matmul_aswt -DCATLASS_ARCH=3510
./output/bin/54_ascend950_fp4_mx_matmul_aswt 256 512 1024 0
# 可执行文件名 |矩阵m轴|n轴|k轴|Device ID
# Device ID可选，默认为0
```

执行结果如下，说明精度比对成功。

```cpp
Compare success.
```

## 使用说明

1、 `gen_data.py`的输入支持trans_a和trans_b，但54_ascend950_fp4_mx_matmul可执行文件不支持，仅仅是trans_a为0及trans_b为1的example示例。

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
A、B支持数据类型为float4_e1m2或float4_e2m1
MxScaleA、MxScaleB支持数据类型为float8_e8m0

其中对于MxScaleA、MxScaleB的数据排布要求如下：
当A为RowMajor时，MxScaleA的shape为（m, ceil(k/64), 2）
当A为ColumnMajor时，MxScaleA的shape为（ceil(k/64), m, 2）
当B为RowMajor时，MxScaleB的shape为（ceil(k/64), n, 2）
当B为ColumnMajor时，MxScaleB的shape为（n, ceil(k/64), 2）

3、 `MxMatmulTla` 与 `BlockMmadTla` 搭配使用的 DispatchPolicy 为 **`Gemm::MmadMx`**（定义见 `include/catlass/gemm/dispatch_policy.hpp`），模板参数顺序与默认值如下：

| 模板参数             | 默认值  | 参数说明                                                                                                            |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `ArchTag`            | 无      | 架构标签，例如 `Arch::Ascend950`                                                                                    |
| `ENABLE_UNIT_FLAG`   | `false` | 是否开启 UnitFlag；当 `L0C_STAGES > 1`（L0C 多缓冲）时必须为 `false`                                                |
| `L1_SCALE_FACTOR_K`  | `16`    | GM→L1 的 MX scale 一次驻留所覆盖的 **L1 K 方向条带个数**；为 `1` 时表示每个 L1 K 条带各搬一次 scale（见类型内注释） |
| `L0C_STAGES`         | `1`     | L0C 缓冲段数；设为 `2` 可开启 L0C 双缓冲（需与 `ENABLE_UNIT_FLAG` 约束一致）                                        |
| `ENABLE_L1_RESIDENT` | `false` | 是否开启 L1 常驻                                                                                                    |
| `L1A_STAGES`         | `2`     | L1 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L1B_STAGES`         | `2`     | L1 上加载矩阵 B 的 buffer 数量                                                                                      |
| `L0A_STAGES`         | `2`     | L0 上加载矩阵 A 的 buffer 数量                                                                                      |
| `L0B_STAGES`         | `2`     | L0 上加载矩阵 B 的 buffer 数量                                                                                      |

设矩阵Shape为`M N K`, L1上的分块大小为`m1 n1 k1`，M方向的分块数量`mTiles = CeilDiv(M, m1)`，N方向的分块数量`nTiles = CeilDiv(N, n1)`，总任务数为`taskBlocks = mTiles * nTiles`，在以下两种情况下可以选择开启enableL1Resident：

1.`mTiles = 1`，且`nTiles > CoreNum`，且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`n1`设置为原来的一半。

2.`nTiles = 1`，且`mTiles > CoreNum`, 且`K < 2 * k1`。此时还可以设置`l0CStages=2`(需要关闭enableUnitFlag)，如果空间不足无法设置`l0CStages=2`，则将`m1`设置为原来的一半。
