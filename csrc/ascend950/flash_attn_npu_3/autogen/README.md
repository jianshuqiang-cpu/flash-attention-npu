# Ascend 950 FA3 编译单元拆分

原来每个 dtype/layout 单元都包含 FP16/BF16 和 FP8 实现头文件，并在一个单元中
实例化 32 个 FP16/BF16 kernel 入口。DN/MHA D128 特化、FD、融合组与跨任务流水
增加了各入口内部的编译工作，九个 dtype/layout 单元中的六个因此成为编译长尾。

现在保留原 `launch_fwd_impl<DType, Format>` 接口，将 FP16/BF16 的重模板进一步拆开：

| 单元 | 数量 | 职责 |
|---|---:|---|
| `fwd_dispatch_{fp16,bf16}_{layout}.cpp` | 6 | 轻量 host 路由，选择 path/softcap |
| `fwd_variant_{dtype}_{layout}_{path}_sc{0,1}.cpp` | 48 | 每个实例化 4 个 LSE/cache 组合 |
| `fwd_dispatch_fp8_{layout}.cpp` | 3 | 独立 FP8 实现，保留原有 24 个组合 |

path 为 `nomask`、`dn`、`causal`、`swa`。只有 No mask 使用 DN 分支；masked 场景
忽略 `enable_dn`，与原分发一致。softcap 在每个叶子单元中固定，LSE 和 cache 模式
仍由叶子 launcher 选择。head dimension、序列长度继续使用 runtime tiling。

生成的 `fwd_dispatch_variants.hpp` 声明全部 48 个 extern template，轻量路由不包含
kernel 实现。`fwd_dispatch_variant_impl.hpp` 只包含 FP16/BF16 实现；
`fwd_dispatch_quant_impl.hpp` 只包含 FP8 实现。两者都不再包含 `torch/extension.h`。
原来的 192 个 FP16/BF16 入口和 72 个 FP8 入口、模板参数及 kernel 参数列表保持一致。
数值计算、tiling 和设备侧流水未在本次调整。

Backward 同样使用六个轻量 dtype/layout 路由，48 个
`bwd_variant_{dtype}_{layout}_{mask}_{dtm}_sc{0,1}.cpp` 叶子单元，每个仅实例化一个
`FlashAttentionV3Bwd950`。`mha_bwd.cpp` 只准备参数和 tiling，不再通过
`flash_api.cpp` 把全部 48 个 backward kernel 集中实例化。布局值由静态断言校验。

## 生成与构建

在仓库根目录执行：

```bash
python csrc/ascend950/flash_attn_npu_3/autogen/generate_kernels.py
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
FLASH_ATTN_BUILD_NPU=950 FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 \
python setup.py build_ext --inplace
```

生成器仅在内容变化时写文件，重复执行不改变 mtime，避免破坏增量构建。它只清理带有
自身生成标记的过期 `fwd_variant_*.cpp`，保留手写文件。源码分发包继续包含生成的
`.cpp/.hpp`，构建时不需要重新运行生成器。

`setup.py` 自动收集 57 个 forward 和 54 个 backward autogen 单元，加上
`flash_api.cpp` 和 `combine_kernel.cpp`，共 113 个 ASC 编译单元；AICPU metadata 单独编译。
`build_ext --parallel N` 优先于 `MAX_JOBS=N`；未设置时取逻辑 CPU 数、affinity 和
继承的 cgroup-v2 CPU 配额的最小值，再以单元数封顶。不支持 cgroup-v2 的环境使用
CPU 数/affinity。比如本机显示 256 个逻辑 CPU，但当前用户配额只有 16 核，默认用 16。
较小机器应按内存和可用核数降低并发。日志中的 `[compile-done]` 给出逐单元耗时，
输出加锁以避免多个编译线程把命令行写到一起。

## 验证

`tests/test_flash_attn_npu_v3_950_autogen.py` 检查完整生成矩阵、内容与 mtime 幂等性、
过期文件清理、384 组 forward 路由和 48 组 backward 路由组合。
完整构建后还比较两版共享库中的 kernel 入口集合，并运行 FP16/BF16、FP8 回归。

本次编译耗时、冷编译日志、增量构建结果及测试记录归档在
`JD/20261111/compile_autogen/`，实测结果见该目录的 `analysis.md`。
