# 体系结构与分层设计

## 总览

```text
┌─────────────────────────────────────────────────────────────┐
│                  Python 层 (torch_catlass)                    │
│  __init__.py  │  ops/basic_matmul.py  │  ops/quant_matmul.py │
│  _load_kernel_libs()  │  _load_main_lib()  │  get_npu_arch()  │
└──────────────────────────┬──────────────────────────────────┘
                           │ torch.ops.catlass.*
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                C++ 桥接层 (src/)                              │
│  register.h ── TORCH_LIBRARY / REGISTER_TORCH_FUNC           │
│  run_npu_func.h ── OpCommand 包装器                          │
│  workspace.cpp ── 基于 torch 的 NPU 分配器引导               │
│  template/matmul.h / quant_matmul.h ── KernelFunc 包装器     │
└──────────────────────────┬──────────────────────────────────┘
                           │ JitCompiler::getKernel()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               JIT 编译层 (kernels/)                           │
│                                                              │
│  ┌─────────────────┐    ┌───────────────────────────────┐   │
│  │  jit_compiler    │    │  jit_macro_generator          │   │
│  │  · 缓存(内存/磁盘)│    │  · generate() → MacroMap     │   │
│  │  · SHA256 UUID   │    │  · appendTo()                 │   │
│  │  · bisheng 调起   │    └───────────────────────────────┘   │
│  └─────────────────┘                                         │
│  ┌─────────────────┐    ┌───────────────────────────────┐   │
│  │  jit_config      │    │  jit_{logger,util,macros}    │   │
│  │  · 环境变量      │    │  · 日志 / 进程执行            │   │
│  │  · 编译器标志    │    │  · 架构检测 / 包含路径        │   │
│  └─────────────────┘    └───────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  内核模板 (如 basic_matmul_impl.cpp)                  │   │
│  │  运行时由 bisheng 编译 → .so → dlopen                  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ dlsym("run")
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              内核公共层 (kernels/common/)                     │
│  kernel_runner.h ── RunKernel<Kernel>() 宿主编译启动器      │
│  workspace_alloc.h ── 全局 workspace 分配器注入              │
│  common.h ── IsNeedPadding() 重载                            │
│  optimized_macro_generator.h ── ApplyOptMacros()             │
│  tile_shape_scaler.h ── 元素类型感知的 tile shape 缩放       │
└─────────────────────────────────────────────────────────────┘
```

## 层次职责

| 层次           | 职责                                                                          | 关键文件                                                                         |
| -------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Python 层**  | 用户 API 入口、动态库加载、架构识别、版本管理                                 | `torch_catlass/__init__.py`, `torch_catlass/ops/*.py`                            |
| **C++ 桥接层** | `torch.ops.catlass.*` 注册、NPU OpCommand 分发、Tensor → kernel ABI 转换      | `src/common/register.h`, `src/common/run_npu_func.h`, `src/include/template/*.h` |
| **JIT 编译层** | 运行时编译 Ascend C 模板、磁盘/内存缓存、SHA256 UUID、宏生成                  | `kernels/jit/*.cpp`, `kernels/include/jit_*.h`                                   |
| **内核公共层** | JIT 模板与预编译内核共享的工具：padding 判定、kernel 启动器、workspace 分配器 | `kernels/common/*.h`                                                             |
| **构建系统**   | 多架构预编译内核、JIT 模板构建期验证、wheel 打包、版本管理                    | `CMakeLists.txt`, `build.sh`, `pyproject.toml`                                   |

## 设计原则

1. **JIT 优先**：新算子默认走 JIT 编译路径。dtype、layout、shape 变化产生不同的 SHA256 UUID，缓存独立的 `.so`，无需重新链接。

2. **SHA256 UUID**：每个内核特化由所有编译宏（按 key 排序）+ 架构 + 内核类型共同确定 SHA256 哈希值。确定性、无冲突、无文件名长度问题。

3. **Workspace 注入**：NPU 临时内存由 torch 层通过 `g_catlassWorkspaceAlloc` 全局函数指针统一管理。JIT 模板通过 `RTLD_GLOBAL` 共享此符号，无需显式传参。

4. **ABI 稳定**：所有 JIT 编译的内核入口统一为 3 参数 `(blockNum, stream, params)`，`params` 为不透明指针。类型相关数据封装在 `MatmulParams` 结构体中。

5. **预编译回退**：性能关键或参数固定的内核可提前编译为架构特定的 `.so`，运行时直接 `dlopen`，绕过 JIT 编译开销。
