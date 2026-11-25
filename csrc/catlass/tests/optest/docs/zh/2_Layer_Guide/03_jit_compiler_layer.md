# JIT 编译层

JIT 编译层在运行时编译 Ascend C 内核模板，并将编译产物 `.so` 缓存到磁盘，跨进程复用。

## 文件布局

```text
kernels/
├── include/
│   ├── jit_compiler.h          # JitCompiler 单例（公开 API）
│   ├── jit_config.h            # 编译器标志、环境变量名、JitKernelType
│   ├── jit_logger.h            # 日志宏 (JIT_LOG, JIT_LOGE)
│   ├── jit_macros.h            # JIT_CHECK, JIT_THROW 宏
│   ├── jit_macro_generator.h   # JitMacroGenerator<T> 模板 + TParams 特化
│   ├── jit_sha256.h            # 自包含 SHA256 实现
│   └── jit_util.h              # MacroMap、架构检测、编译器路径解析
└── jit/
    ├── jit_compiler.cpp         # JitCompiler 实现
    ├── jit_macro_generator.cpp  # TParams 宏生成
    ├── jit_logger.cpp           # 从环境变量读取日志级别
    ├── jit_util.cpp             # 架构/编译器/模板路径检测
    └── workspace_alloc.cpp      # 全局 workspace 分配器符号
```

## JIT 编译管线

```text
用户代码
   │
   ├── 填充 TParams + MatmulParams
   │
   ├── JitMacroGenerator<TParams>::generate(name, tParams)
   │     → MacroMap { CATLASS_JIT_ELEMENT_A=half, CATLASS_JIT_LAYOUT_A=RowMajor, ... }
   │
   ├── ApplyOptMacros(macros, m, n, k, isNzA, transA, isNzB, transB)
   │     → 添加 CATLASS_JIT_NEED_PADDING_A/B, CATLASS_JIT_BLOCK_SCHEDULER
   │
   ├── JitCompiler::instance().getKernel(templatePath, macros, kt)
   │     │
   │     ├── makeKernelUuid(macros)
   │     │     ├── 排序所有 (key, value) 对
   │     │     ├── 添加 __ARCH__ + __KT__
   │     │     ├── 拼接 "key=val&key=val&..."
   │     │     └── SHA256 → 64 字符 hex UUID
   │     │
   │     ├── 检查内存缓存 (loaded_ 映射)
   │     ├── 检查磁盘缓存 ({cacheDir}/{uuid}.so)
   │     ├── 未命中：compile(name, templatePath, macros, kt, soPath)
   │     │     ├── buildCompilerArgs(...)
   │     │     │     ├── bisheng 路径
   │     │     │     ├── -x asc, -std=c++17, -O2, -shared
   │     │     │     ├── --npu-arch=dav-{arch}, -DCATLASS_ARCH={arch}
   │     │     │     ├── -DKERNEL_TYPE=__cube__|__vector__|__mix__
   │     │     │     ├── -DKERNEL_NAME={name}
   │     │     │     ├── -DCATLASS_JIT_KERNEL_NAME={name}_arch{arch}
   │     │     │     ├── 所有用户宏作为 -D{key}={value}
   │     │     │     ├── 环境变量中的包含路径
   │     │     │     ├── 模板源码路径
   │     │     │     └── -o {soPath}
   │     │     │
    │     │     └── RunProcessCapture(args) → bisheng 编译模板 → .so
    │     │          所有参数通过单引号包裹传入 shell，防止 __mix__(1,2)
    │     │          等特殊字符被 /bin/sh 误解析。
   │     │
   │     ├── dlopen(soPath)
   │     └── dlsym("run") → JitEntryFn
   │
   ├── entry(blockNum, stream, &params)
   │     └── 调用 JIT 编译的内核
   │
   └── aclrtSynchronizeStream(stream)
```

## 内核 UUID 生成 (jit_compiler.cpp)

```text
makeKernelUuid(macros):
  1. 收集 MacroMap 中所有宏
  2. 添加 "__ARCH__" = npuArch_ (如 "2201")
  3. 添加 "__KT__" = kt 字符串 (如 "0" 对应 AIC)
  4. 按 key 字典序排序所有键值对
  5. 拼接: "CATLASS_JIT_ELEMENT_A=half&CATLASS_JIT_ELEMENT_B=half&...__ARCH__=2201&__KT__=0"
  6. SHA256 → 64 字符 hex 字符串
```

UUID 同时作为：

- 内存缓存键（`loaded_` 映射）
- 磁盘文件名：`{cacheDir}/{uuid}.so`

SHA256 确保确定性、无冲突的标识符，不受宏内容长度或特殊字符影响。

## JIT 内核 ABI

```cpp
using JitEntryFn = void (*)(uint32_t blockNum, aclrtStream stream, const void* params);
```

所有 JIT 编译的模板导出 C 链接的 `run` 符号：

```cpp
extern "C" void run(uint32_t blockNum, aclrtStream stream,
                    const CatlassKernel::MatmulParams* params);
```

- **blockNum**: 使用的 NPU AI 核数
- **stream**: NPU 异步执行流
- **params**: 不透明参数结构体（类型特定，定义在 `catlass_kernel.h`）

## Shell 转义 (jit_util.cpp)

`RunProcessCapture()` 将编译器参数拼接后通过 `popen()` 传递给 `/bin/sh`。为防止 shell 误解析特殊字符，每个参数均通过 `shellQuote()` 包裹单引号。这对 `-DKERNEL_TYPE=__mix__(1,2)` 中的括号至关重要，否则会导致 `Syntax error: "(" unexpected` 编译失败。

## 配置与标志 (jit_config.h)

环境变量：

| 变量                      | 用途                                                 | 可接受值                      | 默认值                       |
| ------------------------- | ---------------------------------------------------- | ----------------------------- | ---------------------------- |
| `CATLASS_JIT_LOG_LEVEL`   | 日志级别                                             | `0`=None, `1`=Info, `2`=Debug | `0`                          |
| `TORCH_CATLASS_CACHE_DIR` | JIT 磁盘缓存目录                                     | 绝对路径                      | `~/.cache/catlass/jit_cache` |
| `MS_SANITIZE_MEMORY`      | 启用 Ascend 内存消毒器 (`--cce-enable-sanitizer`)    | `1`                           | —                            |
| `TORCH_CATLASS_VERSION`   | 版本字符串注入 `-DCATLASS_VERSION_FULL`              | 包内自动设置                  | "unknown"                    |
| `ASCEND_HOME_PATH`        | CANN 安装根目录，查找 `ccec` 编译器和 runtime 库     | 绝对路径                      | 必设                         |
| `TORCH_CATLASS_PKG_DIR`   | 包安装目录，JIT 依据此路径定位 include 和模板        | 包内自动设置                  | —                            |
| `CATLASS_JIT_AIC_AS_MIX`  | 强制 AIC 发射 `__mix__(1,0)` 替代默认 `__cube__`     | 任意非空                      | —                            |
| `CATLASS_JIT_AIV_AS_MIX`  | 强制 AIV 发射 `__mix__(0,1)` 替代默认 `__vector__`   | 任意非空                      | —                            |
| `CATLASS_JIT_MIX_CV_11`   | 强制 MIX 发射 `__mix__(1,1)` 替代默认 `__mix__(1,2)` | 任意非空                      | —                            |

环境变量分为两类：

- **外部配置**：`ASCEND_HOME_PATH`、`TORCH_CATLASS_CACHE_DIR`、`CATLASS_JIT_LOG_LEVEL`、`MS_SANITIZE_MEMORY`、`CATLASS_JIT_*_AS_MIX` — 用户按需设置。
- **包内注入**：`TORCH_CATLASS_VERSION`、`TORCH_CATLASS_PKG_DIR` — Python loader 在 import 时自动设置。

`JitKernelType` 枚举：

| 值    | 编译器标志                   | 描述               |
| ----- | ---------------------------- | ------------------ |
| `AIC` | `-DKERNEL_TYPE=__cube__`     | 纯 Cube 内核       |
| `AIV` | `-DKERNEL_TYPE=__vector__`   | 纯 Vector 内核     |
| `MIX` | `-DKERNEL_TYPE=__mix__(1,2)` | Cube + Vector 混合 |

## 宏生成 (jit_macro_generator.h / .cpp)

`JitMacroGenerator<TParams>` 从 `TParams` 生成宏：

```text
appendTo(macros, tParams):
  ├── CATLASS_JIT_ELEMENT_{A|B|C|D} = dtype 字符串
  ├── CATLASS_JIT_LAYOUT_{A|B} = layout 字符串（RowMajor/ColumnMajor/zN/nZ）
  └── CATLASS_JIT_LAYOUT_C = "RowMajor"（始终）
```

部分与OptimizedMatmul相关的运行时派生的宏由 `ApplyOptMacros()` 添加：

```text
ApplyOptMacros(macros, m, n, k, isNzA, transA, isNzB, transB):
  ├── CATLASS_JIT_NEED_PADDING_A = "0" | "1"
  ├── CATLASS_JIT_NEED_PADDING_B = "0" | "1"
  └── CATLASS_JIT_BLOCK_SCHEDULER = "BlockScheduler30" | "BlockScheduler31"
```

## SHA256 实现 (jit_sha256.h)

自包含、仅头文件的 SHA256 实现，无外部依赖。

```cpp
std::string digest = Sha256::hash(input_string);  // 返回 64 字符小写 hex
```

专用于内核 UUID 生成。
