# 内核公共层

JIT 编译模板和预编译内核共享的头文件，位于 `kernels/common/`。

这些头文件在构建时通过顶层 `CMakeLists.txt` 安装到 `jit/common/`，通过 `-I{pkgDir}/jit` 包含路径供 JIT 运行时编译使用。

## JIT 模板公约

**JIT 模板必须使用 `common/kernel_runner.h` 中的 `Catlass::RunKernel<Kernel>()` 启动内核，禁止引入 `catlass/gemm/device/device_gemm.hpp`。**`kernel_runner.h` 方案提供自包含的宿主侧启动逻辑，集成了 workspace 分配和 `<<<>>>` NPU 内核调度。

## 文件布局

```text
kernels/common/
├── common.h                    # IsNeedPadding() 重载
├── kernel_runner.h             # RunKernel<Kernel>() 宿主编译启动器
├── optimized_macro_generator.h # ApplyOptMacros() 形状相关 JIT 宏
├── tile_shape_scaler.h         # 元素类型 → tile shape 缩放
└── workspace_alloc.h           # 全局 workspace 分配器函数指针
```

## Padding 检测 (common.h)

`IsNeedPadding(layout, align)` 判断矩阵是否需要 padding：

| 布局          | 条件                                              | 步幅检查         |
| ------------- | ------------------------------------------------- | ---------------- |
| `RowMajor`    | stride(0) < 65536 ? stride(0) % align != 0 : true | stride(0) = cols |
| `ColumnMajor` | stride(1) < 65536 ? stride(1) % align != 0 : true | stride(1) = rows |
| `zN`          | 始终 false（数据搬运层处理 padding）              | —                |
| `nZ`          | 始终 false（数据搬运层处理 padding）              | —                |

阈值 65536 和对齐参数 `align`（默认 256）以元素单位计。

## 内核启动器 (kernel_runner.h)

`RunKernel<Kernel>(arguments, stream, coreNum)` 是自包含的宿主编译启动器：

```cpp
RunKernel<Kernel>(args, stream, coreNum):
  1. Kernel::CanImplement(args)  ──→ 不支持则提前返回
  2. Kernel::GetWorkspaceSize(args) ──→ workspace 字节数
  3. 如果 wsSize > 0：
       g_catlassWorkspaceAlloc ? 调用它 : aclrtMalloc 回退
  4. Kernel::ToUnderlyingArguments(args, workspace) ──→ Params 结构体
  5. KERNEL_NAME<<<coreNum, nullptr, stream>>>(params)  ← NPU 内核启动
```

`KERNEL_NAME` 和 `KERNEL_TYPE` 宏由 JIT 编译器设置：

- `KERNEL_NAME` = 内核名（如 `"BasicMatmul"`）
- `KERNEL_TYPE` = `__cube__` / `__vector__` / `__mix__(...)`

## Workspace 分配器 (workspace_alloc.h)

NPU 内存分配的全局函数指针接口：

```cpp
using WorkspaceAllocFn = uint8_t* (*)(size_t size);
extern WorkspaceAllocFn g_catlassWorkspaceAlloc;
void CatlassSetWorkspaceAlloc(WorkspaceAllocFn alloc);
```

- 以 `extern "C"` 定义以实现跨库符号可见性
- `g_catlassWorkspaceAlloc` 位于 `libcatlass_kernel_jit_compiler.so`
- 以 `RTLD_GLOBAL` 加载，对所有 JIT `.so` 和 torch 层可见
- 由 torch 层在库加载时通过 `workspace.cpp` 设置
- 未设置时回退到 `aclrtMalloc`（裸 NPU 内存）

## 优化宏生成器 (optimized_macro_generator.h)

`ApplyOptMacros()` 从原始形状/layout 参数计算形状相关的宏：

```cpp
ApplyOptMacros(macros, m, n, k, isNzA, isTransA, isNzB, isTransB, align=256);
```

生成的宏：

| 宏                            | 值                                      | 描述                             |
| ----------------------------- | --------------------------------------- | -------------------------------- |
| `CATLASS_JIT_NEED_PADDING_A`  | "0" / "1"                               | 矩阵 A 是否需要 padding          |
| `CATLASS_JIT_NEED_PADDING_B`  | "0" / "1"                               | 矩阵 B 是否需要 padding          |
| `CATLASS_JIT_BLOCK_SCHEDULER` | `BlockScheduler30` / `BlockScheduler31` | 调度器变体 (m > n → 30, 否则 31) |

Padding 逻辑委托给 `common.h::IsNeedPadding`：

```text
RowMajor:  RowMajor(rows, cols)  → stride(0) = cols
ColumnMajor: ColumnMajor(rows, cols) → stride(1) = rows
```

## Tile Shape 缩放器 (tile_shape_scaler.h)

根据元素类型宽度缩放 GEMM tile shape：

```cpp
using ScaledTile = TileShapeScaler<ElementA, half, GemmShape<128, 256, 256>>::type;
```

`ElementA` 为 `int8_t`（1 字节，half 为 2 字节）时 K 维翻倍：`GemmShape<128, 256, 512>`。`ElementA` 为 `float`（4 字节）时 K 维减半。
