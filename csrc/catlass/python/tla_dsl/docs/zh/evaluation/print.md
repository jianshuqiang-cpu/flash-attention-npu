# 在 CATLASS DSL 中使用打印调试接口

CATLASS DSL 提供 `tla.print` 接口，用于在 kernel 内部进行调试打印。`tla.print` 有两类用法：

| 用法        | 形态                                                     | 说明                                 |
| ----------- | -------------------------------------------------------- | ------------------------------------ |
| 标量打印    | `tla.print(value)` / `tla.print(fmt, *args)`             | 打印单个标量、纯字符串或格式化字符串 |
| Tensor 打印 | `tla.print(tensor)` / `tla.print(tensor, length)`        | 打印整个 Tensor 或其物理前缀         |

`tla.print` 只接受位置参数。Tensor 打印中的 `length` 表示元素个数而不是字节数；静态 shape 的 Tensor 可以省略 `length`，动态 shape 的 Tensor 必须显式传入 `length`。

## 使用示例

下面以 `python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py` 中的 `basic_mmad_kernel` 为例。调试时请在下面两种方案中**二选一**。

### 标量打印

标量打印适合查看运行时整数、浮点数、block 调度和执行流。使用本方案时，kernel 中不能再有 `tla.print(tensor, ...)` 调用。

```diff
# python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(123)  # 打印常量
+       tla.print("Hello world")  # 打印字符串
        tla.set_flag(l1a0_available)
        ...

        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
+           tla.print(
+               "tile={} row={} col={}", block_linear, block_row, block_col
+           )  # 打印格式化字符串
            ...
            for k_l1 in k_l1_range:
                ...
                for k_l0 in k_l0_range:
                    ...
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
+                   if k_l1 == 0 and k_l0 == 0:
+                       tla.print("first mmad done")
```

本方案展示了三种打印：

1. `tla.print(123)`：打印单个标量。
2. `tla.print("Hello world")`：打印纯字符串。
3. `tla.print("tile={} ...", ...)`：格式化打印 `block_linear`、`block_row` 和 `block_col` 等运行时值。

### Tensor 打印

Tensor 打印适合查看 GM 或 UB 中的数据。

```diff
# python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    with tla.cube():
+       tla.print(gm_a, 256)
        tla.set_flag(l1a0_available)
        ...
```

`tla.print(gm_a, 256)` 打印 GM 输入 A 从有效地址开始的连续 256 个元素。使用 Basic MMAD 的默认参数时，`gm_a` 的 dtype 为 `f16`，shape 为 `[256, 1024]`。

### Tensor 打印支持范围

- 只支持 rank-1 或 rank-2 Tensor。
- 支持 `f16`、`f32`、`i8`、`i16`、`i32`、`u8`、`u16` 和 `u32`。
- GM Tensor 可以在 `tla.cube()` 或 `tla.vector()` 区域中打印；UB Tensor 只能在 `tla.vector()` 区域中打印。
- `length` 可以是静态整数或运行时整数 SSA 值。静态值必须在 1～262112 之间，且不能超过 Tensor 的元素数。
- 打印的是有效地址开始的连续物理前缀。Tensor 的逻辑 shape 只用于展示，接口不会按照 stride 收集元素，也不会重排 packed layout。
- `tla.print` 不会为 UB Tensor 自动插入生产者同步；打印前必须确保相关搬运或计算已经完成。

## 编译运行

确认 kernel 中只保留上述一种打印方案，并且没有残留的 L1、L0A、L0B 或 L0C Tensor 打印后再运行。CATLASS DSL 编译器会根据 kernel 中的打印类型选择对应的调试 FIFO 通路，无需额外编译选项。建议启用强制重新编译环境变量，确保修改后的 kernel 和打印辅助代码重新生成：

```bash
cd python/tla_dsl/examples/end_to_end/basic_mmad

# 单 block 启动方便观察输出
CATLASS_DSL_FORCE_RECOMPILE=1 python basic_matmul.py --device 0 --block-num 1
```

## 输出示例

### 标量打印输出

标量、纯字符串和格式化打印均以 `TLA printf:` 为前缀：

```text
TLA printf: core=0 block=0 x=123
TLA printf: core=0 block=0 Hello world
TLA printf: core=0 block=0 tile=0 row=0 col=0
TLA printf: core=0 block=0 first mmad done
...
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

### Tensor 打印输出

使用 Basic MMAD 的默认 shape 和默认 `f16` 配置时，Tensor FIFO 正常解析后会输出数据类型、shape、打印元素数和元素值：

```text
tla.print dtype=f16 shape=[256,1024] count=256 values=[-1.66015625, -2.26953125, 3.0703125, 0.9375, ...]
passed=True cache_key=<cache-key>
kernel.o=<cache-dir>/kernel.o
```

实际元素值、`cache_key` 和产物路径以运行环境为准。使用多个 block 启动时，不同核之间的打印顺序也可能不同。

## 注意事项

- `tla.print` 只能在 `tla.cube()` 或 `tla.vector()` 区域内使用。
- 一个 kernel 中只能使用标量打印或 Tensor 打印中的一种；纯字符串和格式化字符串均属于标量打印。
- `tla.print` 可能明显影响 kernel 性能，建议仅在调试阶段使用，正式运行时删除所有调测代码。
- Tensor 打印只接受 GM 或 UB Tensor，暂不支持打印 L1、L0A、L0B、L0C 等 buffer 中的数值。
- 在大 shape 精度调试时，可使用 `--block-num 1` 减少并发打印核数，避免不同 block 的输出相互交错。它不保证减少 tile 内打印的总行数。
- 打印调试接口的底层实现调用 AscendC 调测接口，可阅读以下文档获取更多信息：
  - [AscendC::DumpTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E8%B0%83%E8%AF%95%E6%8E%A5%E5%8F%A3/%E4%B8%8A%E6%9D%BF%E6%89%93%E5%8D%B0/DumpTensor.md)
  - [AscendC::printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E8%B0%83%E8%AF%95%E6%8E%A5%E5%8F%A3/%E4%B8%8A%E6%9D%BF%E6%89%93%E5%8D%B0/printf.md)
