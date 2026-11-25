# 在TLA DSL样例工程使用调测API

调测API是TLA DSL提供的调试能力，可进行kernel内部的标量/格式化打印、Tensor内容的查看。在 TLA DSL 中，以上能力统一通过 `tla.print` 暴露。`tla.print` 有两类用法：

| 用法 | 形态 | 说明 |
| --- | --- | --- |
| 标量/格式化打印 | `tla.print(value)` / `tla.print(fmt, *args)` | 打印单个标量或格式化字符串 |
| 张量 dump（`tla.dump_tensor` 等价用法） | `tla.print(tensor, length=None)` | 打印 GM 或 AIV 上 UB 张量的物理前缀 |

## 使用示例

下面以 `batched_matmul`为例，演示基于调测API的测试过程。

### 插入调试代码

在想进行调试的层级，增加调测API调用，如在`python/tla_dsl/examples/end_to_end/batched_matmul/batched_matmul.py`的`batched_matmul_kernel`核函数中添加下述代码。

```diff
# python/tla_dsl/examples/end_to_end/batched_matmul/batched_matmul.py
        for loop_idx in block_range:
            batch_idx = loop_idx // mn_blocks
            mn_linear = loop_idx % mn_blocks
+           tla.print("loop={} batch={} mn={}", loop_idx, batch_idx, mn_linear)
            ...
            for k_l1 in k_l1_range:
                ...
                for k_l0 in k_l0_range:
                    ...
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
+                   if k_l1 == 0 and k_l0 == 0:
+                       tla.print("first mmad done")
+                       tla.print(l0_c, 8)
```

上例在三层循环的关键位置插入了三类打印：

1. **格式化标量打印**（`tla.print("loop={} ...", ...)`）—— 输出当前 `loop_idx`、`batch_idx`、`mn_linear`，用于确认 block 调度与 swizzle 路径。
2. **纯字符串打印**（`tla.print("first mmad done")`）—— 作为执行流探针，确认首次 `mmad` 已触发。
3. **张量 dump**（`tla.print(l0_c, 8)`）—— 打印 L0C 累加器前 8 个元素，用于首轮累加值的快速核对。

> ⚠️ 注意事项
> - 格式化打印中**不**支持张量参数，`{}`字段数须与标量参数数严格相等，格式串须为纯 ASCII 且不含嵌入 NUL。
> - 张量 dump 只接受 GM 或 UB 张量，UB 张量须在 `tla.vector()` 区域内。

### 编译运行

1. `tla.print` 调用会在编译期自动使能调试通路，无需传入额外编译选项。

```bash
cd python/tla_dsl/examples/end_to_end/batched_matmul

# 默认：batch=5, m=256, n=512, k=1024 | Device ID（可选，默认 4）
python batched_matmul.py --run --device 4

# 自定义 shape
python batched_matmul.py --run --device 4 --batch 4 --m 256 --n 256 --k 256 --block 8
```

### 输出示例（仅为示例，实际输出可能因硬件和算子实现不同而有所差异）

标量与格式化打印以 `TLA printf:` 前缀输出；张量 dump 以 `tla.print` 前缀输出，含 `dtype`/`subblock`/`shape`/`count`/`values`。

```bash
python batched_matmul.py --run --device 4
--- backend=torch_npu batch=5 m=256 n=512 k=1024 dtype_a=f16 dtype_b=f16 dtype_c=f16 layout_a=row layout_b=row ---
TLA printf: core=0 block=0 loop=0 batch=0 mn=0
TLA printf: core=0 block=0 first mmad done
tla.print dtype=float32 subblock=0 shape=[256,256] count=8 values=[1.234375, -0.562500, 3.125000, 2.000000, -1.500000, 4.000000, 0.875000, -2.250000]
... #每个Cube核都会输出一次信息
compile_ok=True host=torch_npu layout_a=row layout_b=row dtype_a=f16 dtype_b=f16 dtype_c=f16 batch=5 m=256 n=512 k=1024
launch_ok=True
C equals batched golden? True
first mismatch=None
```

## 单独验证张量 dump（`tla.dump_tensor` 等价用法）

在 `tla.vector()` 区域内调用 `tla.print(value, 16)`，即等价于 `tla.dump_tensor`：

```python
@tla.kernel
def print_tensor_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)
```

连续两次 dump 同一张量（不同前缀长度）的写法：

```python
@tla.kernel
def print_tensor_aiv_two_calls_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)
        tla.print(value, 8)
```

### 输出示例（仅为示例，实际输出可能因硬件和算子实现不同而有所差异）

```text
tla.print dtype=float32 subblock=0 shape=[8,4] count=16 values=[0.0, -0.0, 1.0, -2.5, nan, inf, -inf, 3.25, 0.0, -0.0, 1.0, -2.5, nan, inf, -inf, 3.25]
compile_ok=True
launch_ok=True
output_ok=True
```
