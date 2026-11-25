# 构建系统

构建系统基于 scikit-build-core（CMake + Python wheel），支持 NPU 内核的多架构构建。

## 必需环境

| 变量               | 用途                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| `ASCEND_HOME_PATH` | CANN 安装根目录，须指向包含 compiler 和 runtime 的有效 Ascend 工具包 |

## 构建入口

```bash
bash build.sh                    # 推荐的用户构建脚本
uv build                         # PEP 517 构建
python -m pip wheel .            # 直接 wheel 构建
```

## 构建流程 (build.sh)

```text
build.sh
  │
  ├── 1. 版本检测
  │     └── git describe → PEP 440 版本 → _version.py
  │
  ├── 2. scikit-build-core
  │     └── cmake 配置 + 编译 + 安装
  │           │
  │           ├── CMakeLists.txt (根)
  │           │     ├── find_package(ASC)  ← Ascend C 工具链
  │           │     ├── find_package(Torch)
  │           │     ├── add_subdirectory(kernels)
  │           │     ├── add_subdirectory(utils)
  │           │     └── add_subdirectory(src)
  │           │
  │           ├── kernels/CMakeLists.txt
  │           │     ├── catlass_kernel_jit_compiler (共享库)
  │           │     ├── add_kernel() 宏（每个算子）
  │           │     ├── catlass_kernel_jit (所有 JIT 入口点)
  │           │     └── jit_verify_template() 模板验证
  │           │
  │           ├── utils/CMakeLists.txt → catlass_torch_utils
  │           │
  │           └── src/CMakeLists.txt → catlass_torch
  │
  └── 3. 安装到 site-packages
```

## 输出布局

```text
site-packages/torch_catlass/
├── __init__.py
├── _version.py
├── ops/
├── lib/
│   ├── libcatlass_torch.so                # torch.ops.catlass 注册
│   ├── libcatlass_kernel_jit_compiler.so  # JIT 编译器单例
│   ├── libcatlass_kernel_jit.so           # JIT 内核入口点
│   └── {arch}/
│       ├── libcatlass_kernel_{arch}_basic_matmul.so
│       └── libcatlass_kernel_{arch}_quant_optimized_matmul_tla.so
└── jit/
    ├── catlass/   # JIT 编译用 CATLASS 头文件
    ├── tla/       # JIT 编译用 TLA 头文件
    └── templates/ # JIT 内核模板源文件
        ├── basic_matmul_impl.cpp
        └── quant_optimized_matmul_tla_impl.cpp
```

## add_kernel() CMake 宏

定义在 `kernels/CMakeLists.txt`，支持两种内核类型：

### JIT 模式

```cmake
add_kernel(
    NAME basic_matmul
    KERNEL_TYPE jit
    NPU_ARCH_LIST 2201 3510
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_impl.cpp
    basic_matmul.cpp
)
```

- entry 源文件收集到 `catlass_kernel_jit` 共享库
- 模板文件安装到 `jit/templates/`
- 公共头文件 (`kernels/common/`) 安装到 `jit/common/`
- `jit_verify_template()` 在构建期验证模板

### 预编译模式

```cmake
add_kernel(
    NAME basic_matmul
    KERNEL_TYPE prebuilt
    NPU_ARCH_LIST 2201 3510
    basic_matmul_impl.cpp
)
```

- 按架构分别编译独立动态库
- 架构特定标志：`--npu-arch=dav-{arch}`、2201 溢出检测等
- 始终构建 `_ms` 变体（内存消毒器）与普通版本
- 安装到 `lib/{arch}/`

## JIT 模板验证 (cmake/jit_verify_template.cmake)

构建期对每个 JIT 模板进行试编译：

```text
jit_verify_template(NAME basic_matmul TEMPLATE basic_matmul_impl.cpp NPU_ARCH_LIST 2201 3510 ...)
  → 对每个架构：OBJECT 库通过 bisheng ASC 编译
```

验证：

- 模板语法正确
- 所需头文件可访问
- 架构特定代码路径可编译
- 构建期发现问题，而非运行时

## 版本管理

版本从 catlass git 仓库在构建时推导：

```bash
git describe --tags --always --dirty
  → v1.5.0-41-g10fb189       # PEP 440: 1.5.0.dev41+g10fb189
  → v1.5.0                   # PEP 440: 1.5.0
  → v1.5.0-dirty             # PEP 440: 1.5.0.dev0+g10fb189
  → (无 tag)                 # PEP 440: 0.0.0+g10fb189
```

版本用于：

1. 写入 `torch_catlass/_version.py`
2. 暴露为 `os.environ["TORCH_CATLASS_VERSION"]`
3. 注入 JIT 内核编译：`-DCATLASS_VERSION_FULL=...`
