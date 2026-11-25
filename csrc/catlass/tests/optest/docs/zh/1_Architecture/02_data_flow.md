# 端到端数据流

本文跟踪一次完整的算子调用，从 Python 到 NPU 内核执行。

## 调用链路

```bash
用户 Python 代码
  │
  │  result = torch_catlass.basic_matmul(A, B, outDType="float16")
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第一层：Python Op 包装器 (torch_catlass/ops/basic_matmul.py)      │
│                                                                  │
│  - 规范化 outDType (str → torch.dtype)                          │
│  - 调用 torch.ops.catlass.basic_matmul(A, B, outDType, ...)     │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ torch.ops.catlass.basic_matmul()
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第二层：C++ Op 注册 (torch.ops 分发到 C++ 实现)                  │
│                                                                  │
│  - TORCH_LIBRARY(catlass, ...) 分发到注册的函数                  │
│  - RUN_NPU_FUNC(basic_matmul, ...) 通过 OpCommand 包装           │
│  - 填充 TParams (dtype, layout 标志)                      │
│  - 填充 MatmulParams (m, n, k, tensor 地址)                     │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ BasicMatmul(blockNum, stream, tParams, params)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第三层：JIT 分发 (kernels/00_basic_matmul/basic_matmul.cpp)      │
│                                                                  │
│  1. JitMacroGenerator<TParams>::generate()                │
│     → MacroMap { CATLASS_JIT_ELEMENT_A, ..., CATLASS_JIT_LAYOUT }│
│                                                                  │
│  2. JitCompiler::instance().getKernel(template, macros, kt)     │
│     ├── makeKernelUuid(macros)                                  │
│     │   ├── 排序所有 (key, value) 对                             │
│     │   ├── 添加 __ARCH__ + __KT__                               │
│     │   ├── 拼接 "key=val&key=val&..."                          │
│     │   └── SHA256 → 64 字符 hex UUID                          │
│     │                                                             │
│     ├── 检查 loaded_ 缓存 → 命中则返回                           │
│     ├── 检查 {cacheDir}/{uuid}.so → 命中则 dlopen 返回           │
│     └── 未命中：                                                 │
│         ├── buildCompilerArgs(...) → bisheng 编译                 │
│         ├── dlopen + dlsym("run")                                │
│         └── 缓存到 loaded_ 映射                                   │
│                                                                  │
│  3. entry(blockNum, stream, &params)                             │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ JIT 编译的 "run" 函数
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第四层：JIT 内核模板 (basic_matmul_impl.cpp)                      │
│                                                                  │
│  extern "C" void run(blockNum, stream, params):                 │
│    - 从 params 创建 GemmCoord{m, n, k}                         │
│    - 构建 MatmulKernel::Arguments{shape, A, B, C}               │
│    - 调用 RunKernel<MatmulKernel>(args, stream, blockNum)       │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ RunKernel<MatmulKernel>(args, stream, coreNum)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ 第五层：内核启动器 (kernels/common/kernel_runner.h)              │
│                                                                  │
│  1. MatmulKernel::CanImplement(args) → 检查支持                  │
│  2. MatmulKernel::GetWorkspaceSize(args) → 计算 workspace 大小  │
│  3. 分配 workspace: g_catlassWorkspaceAlloc(n)  [TorchNPU]     │
│  4. MatmulKernel::ToUnderlyingArguments(args, ws) → Params      │
│  5. <<<coreNum, nullptr, stream>>>(params)  ← NPU 内核启动      │
└──────────────────────────────────────────────────────────────────┘
```

## 数据变换过程

```text
Python:  torch.Tensor (NPU 存储)
   │
   ├── .storage().data()  ──→ 设备指针 (void*)
   ├── .shape()           ──→ m, n, k
   └── .scalar_type()     ──→ dtype
         │
         ▼
TParams:                          MatmulParams:
  element["A"] = aclDataType              m = shape[0]
  element["B"] = aclDataType              n = shape[1]
  element["C"] = aclDataType              k = shape[2]
  trans["A"] = false                      inputAddr[0] = A.data_ptr()
  trans["B"] = false                      inputAddr[1] = B.data_ptr()
  nz["A"] = false                         outputAddr[0] = C.data_ptr()
  nz["B"] = false
         │
         ├── JitMacroGenerator → MacroMap + arch + kt
         │                         │
         │                         └── SHA256 → uuid.so
         │
         └── JIT 内核 "run" 读取 params->inputAddr, params->m/n/k
```

## ABI 约定

```cpp
JitEntryFn = void(*)(uint32_t blockNum, aclrtStream stream, const void* params)

模板侧：
  extern "C" void run(uint32_t blockNum, aclrtStream stream,
                       const CatlassKernel::MatmulParams* params)

包装器侧：
  entry(blockNum, stream, static_cast<const void*>(&params))
```

`params` 在 ABI 层始终为 `MatmulParams*`。类型特定的数据在结构体内部，内核模板按预期类型转换回去。

## 缓存生命周期

```text
进程启动
  │
  ├── JitCompiler::instance() (惰性单例)
  │
  ├── 首次 getKernel("basic_matmul_impl.cpp", macros, AIC)
  │     ├── UUID = SHA256(排序后的宏 + arch + kt)
  │     ├── 磁盘未命中 → bisheng 编译 → {uuid}.so
  │     ├── dlopen → dlsym("run") → 存入 loaded_[uuid]
  │     └── 返回入口函数
  │
  ├── 相同 getKernel() 再次调用
  │     └── 内存命中 → 返回缓存的入口函数
  │
  ├── 不同 shape → 不同宏 → 不同 UUID
  │     └── 磁盘命中 → dlopen → 缓存 → 返回入口函数
  │
  ├── 不同进程 → 不同内存，相同磁盘缓存
  │     └── 磁盘命中 → dlopen (无需重新编译)
  │
  └── 进程退出 → JitCompiler 析构 → clearCache()
        └── dlclose 所有已加载的 .so
```
