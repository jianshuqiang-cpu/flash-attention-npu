# CV (Cube + Vector) 融合后处理类 端到端示例

本目录下提供的系列样例演示 **CATLASS DSL** 下系列后处理类CV（Cube + Vector）融合算子的计算过程。

## 功能说明

后处理类算子实现基础矩阵乘并复合后续计算步骤。

$$
\begin{aligned}
D &= f(A \times B)
\end{aligned}
$$

本目录下样例包含的后处理操作见后续介绍。


## 代码组织

```plain
./basic_mmad_epilogue
├── matmul_add.py
├── matmul_add_ub.py
├── matmul_bias.py
├── matmul_leaky_relu.py
├── matmul_sigmoid.py
├── matmul_silu.py
├── matmul_tanh.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`matmul_add.py`**](matmul_add.py) | 实现 `D = A@B + X` 计算功能。 |
| [**`matmul_add_ub.py`**](matmul_add_ub.py) | 实现 `D = A@B + X` 计算功能，启用L0C -> UB通路将矩阵乘结果搬出到 UB。 |
| [**`matmul_bias.py`**](matmul_bias.py) | 实现 `D = A@B + bias` 计算功能，其中 `bias` 为一维 `(n,)` 的广播向量。 |
| [**`matmul_leaky_relu.py`**](matmul_leaky_relu.py) | 实现 `D = LeakyRelu(A@B)` 计算功能，其中`α` 默认为 `0.1`。 |
| [**`matmul_sigmoid.py`**](matmul_sigmoid.py) | 实现 `D = Sigmoid(A@B)` 计算功能。 |
| [**`matmul_silu.py`**](matmul_silu.py) | 实现 `D = Silu(A@B)` 计算功能。 |
| [**`matmul_tanh.py`**](matmul_tanh.py) | 实现 `D = Tanh(A@B)` 计算功能。 |

## 约束说明

 - 左、右矩阵及结果矩阵所支持的数据组合类型如下。

| 算子 | `DTYPE_A` / `DTYPE_B` | `DTYPE_C` |
|------|-------------------|---------|
| add, bias, leaky_relu, sigmoid, silu | f16 / bf16 / f32 | f16 或 f32 |
| add_ub, tanh | f16 / bf16 / f32 | f32 |


## 使用示例

要运行本路径下的样例，请参考[环境配置](../../../docs/zh/dev_guide/00_environment_setup.md)完成部署。

### 命令行参数

主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `0` | 上板执行使用的 NPU 设备号。 |
| `--m` | `256` | 矩阵乘左矩阵 A 的行数 |
| `--n` | `256` | 矩阵乘右矩阵 B 的列数 |
| `--k` | `256` | 矩阵乘累加轴的大小 |
| `--layout-a` / `--layout-b` | `"row"` / `"row"` | 左、右矩阵 A、B 的数据排布格式，可选 `"row"` 或 `"col"`，表示行优先或列优先布局。 |
| `--dtype-a` / `--dtype-b` | `"f32"` / `"f32"` | 左、右矩阵 A、B 的数据类型，可选范围参考约束说明。 |
| `--dtype-c` | `"f32"` | 结果矩阵 C 的数据类型，可选范围参考约束说明（`add_ub` / `tanh` 仅支持 f32）。 |
| `--block-num` | `-1` | 启用的 AIC 核数，`-1` 表示自动探测可用核数（满核）。 |


### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 其余变体替换文件名即可
python examples/end_to_end/basic_mmad_epilogue/matmul_add.py --device 0 \
  --m 256 --n 256 --k 256 \
  --layout-a row --layout-b row \
  --dtype-a f16 --dtype-b f16 --dtype-c f32
```

默认测试条件下，预期输出：

```text
--- mnk=(256,256,256) layout=row/row dtype=f16/f16/f32 ---
passed=True cache_key=<cache_key>
kernel.o=<cache_dir>/<cache_key>/kernel.o
```

其中 `passed`结果为`True`或`False` 表明 NPU 计算结果与golden参考值精度校验是否通过；`cache_dir` 是指定的缓存目录， `cache_key` 是编译缓存的哈希值。
