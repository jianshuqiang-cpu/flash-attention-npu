# 基础向量运算端到端示例

本目录下提供的系列样例是 **CATLASS DSL** 下的多种向量（Vector）操作示例。


## 功能说明

向量运算是 NPU 上基础的计算原语，在 AIV 物理核上执行，整体执行流程包括：
1. 构造 GM （Global memory）上的输入/输出 `tla.Tensor`；
2. 启动MTE2，将数据搬运至 UB （Unified Buffer）；
3. 加载至寄存器上，以 `VL`（向量寄存器位宽， 256 字节）为粒度分块执行各类向量指令；
4. 输出回 UB，然后启动 MTE3，将数据搬运回 GM。


## 代码组织

本目录组织结构及文件概述如下：

```plain
./vector_ops
├── arange_op.py
├── binary_op.py
├── bitwise_ops.py
├── cast_multi.py
├── compare_mask.py
├── gather_op.py
├── interleave_op.py
├── load_and_store_scalar_after_reduction.py
├── load_dintlv_op.py
├── load_store_mask.py
├── load_us_b8_op.py
├── masked_binary.py
├── multi_binary.py
├── reduction_ops.py
├── register_control_flow.py
├── squeeze_op.py
├── store_pack.py
├── unary_ops.py
├── vector_op_harness.py
└── README.md
```

| 文件 | 概述 |
|------|------|
| [**`vector_op_harness.py`**](vector_op_harness.py) | 公共测试框架，用于执行批量测试，封装命令行参数解析、dtype/shape 配置、输入构造、golden 校验与批量/扫描执行逻辑。 |
| [**`binary_op.py`**](binary_op.py) | 二元运算示例，支持 `add`/`sub`/`mul`/`div`/`max`/`min`，以及非对齐搬运（`add_unalign`）与广播（`add_brc_b32`）变体。 |
| [**`unary_ops.py`**](unary_ops.py) | 一元运算示例，支持 `exp`/`log`/`sqrt`/`abs`/`neg`，以及带掩码的变体。 |
| [**`multi_binary.py`**](multi_binary.py) | 多算子二元计算示例，演示多级混合计算过程。 |
| [**`masked_binary.py`**](masked_binary.py) | 多算子掩码二元运算示例，演示不同掩码模式下 `add`/`sub`/`mul`/`div` 向量计算。 |
| [**`bitwise_ops.py`**](bitwise_ops.py) | 按位运算示例，支持 `bitwise_and`/`bitwise_or`/`bitwise_xor`/`bitwise_not`。 |
| [**`compare_mask.py`**](compare_mask.py) | 比较掩码示例，覆盖 `tla.cmp` 的六种比较模式（lt/le/gt/ge/eq/ne）、以及向量-标量比较。 |
| [**`cast_multi.py`**](cast_multi.py) | `tla.cast` 类型转换示例，覆盖 `f32` / `f16` / `bf16` / `i32` / `i16` / `i8` 之间的多级类型转换。 |
| [**`arange_op.py`**](arange_op.py) | `tla.arange` 序列生成示例，在 UB 上生成递增/递减（`increase`/`decrease`）序列。 |
| [**`gather_op.py`**](gather_op.py) | `tla.gather` 按索引收集示例（独立脚本），依据索引向量从源向量中收集元素。 |
| [**`interleave_op.py`**](interleave_op.py) | 交错运算示例，基于两路输入分别产出交错后的两路输出。 |
| [**`load_dintlv_op.py`**](load_dintlv_op.py) |双目的交织加载示例，单次加载拆出偶数/奇数两路寄存器。 |
| [**`load_us_b8_op.py`**](load_us_b8_op.py) | 上采样加载示例，将 i8 元素上采样至 VL 寄存器（仅 i8）。 |
| [**`load_store_mask.py`**](load_store_mask.py) | 掩码搬运往返示例，掩码 UB 与伴随向量同 dtype。 |
| [**`store_pack.py`**](store_pack.py) | 压缩存储示例，取得低半有效数位做紧凑存储。 |
| [**`reduction_ops.py`**](reduction_ops.py) | `tla.reduce` 归约示例，支持 `add`/`max`/`min` 三种模式，覆盖对齐与非对齐场景。 |
| [**`squeeze_op.py`**](squeeze_op.py) | `tla.squeeze` 示例，按掩码（M4）压缩有效 lane 并紧凑写回。 |
| [**`register_control_flow.py`**](register_control_flow.py) | 控制流示例，演示 for-loop 控制流及 `VectorSSA` 与 `MaskSSA` 跨循环边界存活。 |
| [**`load_and_store_scalar_after_reduction.py`**](load_and_store_scalar_after_reduction.py) | UB 标量访问示例，对归约结果做标量 `load` / `store` 操作。 |


## 约束说明

 - 各样例支持的数据类型如下（`f32`、`f16`、`bf16`、`i8`、`i16`、`i32`）：

| 文件 | 支持的数据类型 |
|------|---------------|
| `binary_op.py` | i8, i16, i32, f16, f32, bf16 |
| `unary_ops.py` | i8, i16, i32, f16, f32, bf16（浮点类运算仅 f16/f32，bf16 不支持） |
| `multi_binary.py` | i32, f32 |
| `masked_binary.py` | i16, i32, f16, f32 |
| `bitwise_ops.py` | f16, bf16, f32, i32, i16, i8 |
| `compare_mask.py` | f32 |
| `cast_multi.py` | f32 |
| `arange_op.py` | i8, i16, i32 |
| `gather_op.py` | f32 |
| `interleave_op.py` | i8, i16, i32, f16, f32, bf16 |
| `load_dintlv_op.py` | f32 |
| `load_us_b8_op.py` | i8 |
| `load_store_mask.py` | f32, f16, i8 |
| `store_pack.py` | i32, i16 |
| `reduction_ops.py` | f32 |
| `squeeze_op.py` | f32, f16, i32 |
| `register_control_flow.py` | f32 |
| `load_and_store_scalar_after_reduction.py` | f32 |

 - 浮点类型（`f32`/`f16`/`bf16`）与整型（`i8`/`i16`/`i32`）在精度校验上采用不同的策略：浮点类型使用容差法进行精度判断（绝对误差可由 `--atol` 给定），整型则进行逐元素比对。


## 使用示例

要运行本路径下的样例，请参考[环境配置](../../../docs/zh/dev_guide/00_environment_setup.md)完成部署。

### 命令行参数

各样例通过公共测试组件提供统一的命令行接口：

```text
<script.py> [op] [--sweep] [--batch-run {op ...}]
            [--device DEVICE] [--block-num BLOCK_NUM]
            [--dtype {f32,f16,bf16,i8,i16,i32}]
            [--shape N] [--shapes N ...] [--sizes N ...]
            [--dtypes ...] [--all-dtypes] [--batch-size {1..4}]
            [--sentinel SENTINEL] [--atol ATOL] [--fail-fast]
```

上述命令行参数具体说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `op` | -（位置参数，必填或与 `--batch-run` 二选一） | 待执行的算子名，取决于待执行的算子名。 |
| `--sweep` | `None` | 全量扫描模式，对 `--dtypes` / `--shapes` / `--sizes` 的组合逐一执行，默认不启用。 |
| `--batch-run` | - | 批量模式，将多个算子打包作多核并发执行。 |
| `--device` | `2` | 上板执行使用的 NPU 设备号。 |
| `--block-num` | 各脚本默认值 | 所启用的 AI Core 核数。 |
| `--dtype` | `f32` | 操作数数据类型。 |
| `--shape` | 各脚本默认向量长度 | 一维扁平向量的元素个数（例如 `--shape 400`）。 |
| `--shapes` / `--sizes` | - | 供 `--sweep` 使用的一维形状列表。 |
| `--dtypes` | 全部支持类型 | 供 `--sweep` 使用的数据类型列表。 |
| `--all-dtypes` | `None` | 对全部支持的数据类型逐一执行，默认不启用。 |
| `--batch-size` | `4` | 批量模式下每个 kernel 打包的算子数（1~4）。 |
| `--sentinel` | 各算子默认哨兵值 | 初始化输出的值。 |
| `--atol` | 各算子默认绝对误差 | 浮点精度校验的绝对误差阈值。 |
| `--fail-fast` | `None` | 扫描/批量模式遇到失败即停止，默认不启用。 |

> `gather_op.py`、`reduction_ops.py`、`load_and_store_scalar_after_reduction.py` 为独立脚本，使用各自的命令行参数（如 `--run`、位置参数 `add/max/min`、`--device` 等）。

### 执行示例

在 `python/tla_dsl` 目录下执行：

```bash
cd python/tla_dsl

# 基础测试（默认 dtype=f32，shape 使用脚本默认值）
python examples/end_to_end/vector_ops/binary_op.py add

# 指定 NPU 设备、数据类型与向量长度
python examples/end_to_end/vector_ops/binary_op.py add --device 1 --dtype f16 --shape 1024

# 全量扫描（多 dtype、多 shape）
python examples/end_to_end/vector_ops/unary_ops.py exp --sweep --dtypes f16 f32 --sizes 31 64 255 1024

# 批量模式（多个算子打包、多核并发）
python examples/end_to_end/vector_ops/binary_op.py --batch-run add sub mul max

# `gather_op.py` 执行（需显式 `--run` ）
python examples/end_to_end/vector_ops/gather_op.py --run --device 0
# `reduction_ops.py` 执行（需显式 `--run` ）
python examples/end_to_end/vector_ops/reduction_ops.py add --run
```

执行测试后，预期输出（以统一框架样例为例）：

```plain
compile_ok=True host=torch_npu op=<op> dtype=<dtype> shape=<shape> layout=row
kernel.o path=<cache_dir>/<cache_key>/kernel.o
launch_ok=True
outputs equal expected <op>? True
first mismatch=None
```

其中 `compile_ok`/`launch_ok` 表示编译与上板启动是否成功；`outputs equal expected <op>?` 为 `True` 或 `False`，表明 NPU 计算结果与 golden 参考值的精度校验是否通过；`kernel.o path` 为编译产物路径。
