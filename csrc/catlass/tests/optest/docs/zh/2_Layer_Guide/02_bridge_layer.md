# C++ 桥接层

桥接层连接 PyTorch 的算子分发系统到 CATLASS JIT 编译管线，位于 `src/`，编译为 `libcatlass_torch.so`。

## 文件布局

```text
src/
├── CMakeLists.txt
├── common/
│   └── workspace.cpp          # 基于 torch 的 NPU 分配器引导
└── include/
    └── common/
        ├── register.h          # TORCH_LIBRARY + REGISTER_TORCH_FUNC
        ├── run_npu_func.h      # OpCommand 分发宏
        └── workspace.h         # (旧版 workspace 接口)
    └── template/
        ├── matmul.h            # MatmulLike<KernelFunc> CRTP 包装器
        └── quant_matmul.h      # QuantMatmulLike<KernelFunc> CRTP 包装器
```

## Op 注册 (register.h)

```cpp
// 在每个算子的 .cpp 中使用：
REGISTER_TORCH_FUNC(basic_matmul);  // 注册 torch.ops.catlass.basic_matmul
```

`REGISTER_TORCH_FUNC` 宏创建静态 `RegisterFunc` 实例：

1. 调用 `GetTorchLibrary()._resolve(name)` 获取规范算子名
2. 通过 PyTorch 的 `inferFunctionSchemaFromFunctor` 从函数签名推断 schema
3. 用 `library.def(schema)` 注册 schema
4. 用 `library.impl()` 绑定实现到 `PrivateUse1` (NPU 后端)

`GetTorchLibrary()` 返回 `thread_local` 单例 `torch::Library`，namespace 固定为 `catlass`。所有算子位于 `torch.ops.catlass.*`。

算子 .cpp 文件遵循此模式：

```cpp
// 1. 声明包装器函数
void BasicMatmul(uint32_t blockNum, aclrtStream stream,
                 const TParams& tParams, const MatmulParams& params);

// 2. 包装为 torch 算子
at::Tensor basic_matmul(at::Tensor mat1, at::Tensor mat2, ...) {
    // 填充 TParams 和 MatmulParams
    // 调用 TemplateWrapper::Run(mat1, mat2, ...)
    //   内部调用 BasicMatmul(blockNum, stream, tParams, params)
}

// 3. 注册
REGISTER_TORCH_FUNC(basic_matmul);
```

## NPU Runtime 分发 (run_npu_func.h)

`RUN_NPU_FUNC` 宏通过 `at_npu::native::OpCommand::RunOpApiV2` 包装内核启动：

```cpp
RUN_NPU_FUNC(func, blockNum, stream, tParams, params)
```

这确保：

- NPU runtime 正确跟踪内核启动上下文
- C++ 异常转换为 `ACL_ERROR_INTERNAL_ERROR`
- 分发前检查函数指针非空

## 模板包装器 (template/matmul.h, template/quant_matmul.h)

`MatmulLike<KernelFunc>` 和 `QuantMatmulLike<KernelFunc>` 封装 matmul 通用流程：

```cpp
GetKernelInfo(mat1, mat2, ...) ──→ 填充 TParams + MatmulParams
       │
       ▼
AllocOutput(tParams, params) ──→ 通过 at::empty 创建输出 tensor
       │
       ▼
Run(mat1, mat2, ...)
  ├── GetKernelInfo(...)
  ├── AllocOutput(...)
  ├── aclrtGetCurrentStream()
  └── KernelFunc(blockNum, stream, tParams, params)  ← 调用 JIT/预编译分发
```

每个包装器处理：

- 从输入 tensor 推导形状（考虑 `transA`/`transB`）
- dtype 转换（torch ↔ acl）
- NPU 上创建输出 tensor
- stream 管理

## Workspace 引导 (workspace.cpp)

在库加载时，静态初始化器注册一个基于 torch 的 NPU 分配器：

```cpp
static WorkspaceBootstrap {
    CatlassSetWorkspaceAlloc(torchWorkspaceAlloc);
} _bootstrap;
```

`torchWorkspaceAlloc(n)` 在 NPU 设备上创建 `n` 个 `torch::kInt8` 元素的 tensor，返回其存储指针。这确保用作 kernel workspace 的 NPU 内存被 PyTorch 分配器正确跟踪。

`CatlassSetWorkspaceAlloc` 设置一个全局函数指针 `g_catlassWorkspaceAlloc`，通过 `RTLD_GLOBAL` 对所有 JIT 编译的内核模板可见。
