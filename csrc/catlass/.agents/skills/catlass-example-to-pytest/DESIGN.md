# Design: catlass-example-to-pytest Skill

## Goal

将 `examples/` 编号算子系统化接入 `tests/optest/`，并保证：

- 接口稳定（ABI 不漂移）
- 参数语义一致（`TParams + Params`）
- 构建运行可验证（build/import/pytest）
- 文档可维护（编号、模块、职责清晰）

## Architecture Mapping

接入目标按四条主线展开：

1. `include/`：先定义 ABI（声明先行）
2. `kernels/`：补充 JIT/prebuilt 实现
3. `src/ + torch_catlass/ops`：完成 torch 注册和 Python 包装
4. `tests/`：建立端到端行为验证

这四条线必须一次闭环完成，避免“只编译不注册”或“只注册无测试”。

## Why Split JIT / Prebuilt

- matmul/gemm/gemv 算子通常依赖模板参数组合，JIT 能避免静态二进制膨胀，且与现有 `JitCompiler` 机制一致。
- conv/flash-attention/mla 这类常见算子的参数面更偏 runtime，prebuilt 模式更直接。

因此技能将其固定分流到：

- `catlass_kernel_jit.h`
- `catlass_kernel_prebuilt.h`

## Parameter Model

### Compile-time (`TParams`)

- dtype/layout/transpose/NZ 等会改变内核模板实例的参数

### Runtime (`Params`)

- shape、batch、group list、buffer address 等运行时数据

### Constraint

- 禁止把编译期 dtype 信息回灌到普通 runtime `Params`，以免 JIT cache key 与调用参数模型脱节。

## Naming Policy

- C++ ABI 导出函数：PascalCase
- Python API：snake_case
- 测试文件：`test_<nn>_<name>.py`
- docstring 必须保留 example 编号（例如 `example 12_quant_matmul`）

## Integration Invariants

1. 头文件声明位置必须与算子类别匹配
2. `src/catlass_torch.cpp` 注册名与 Python wrapper 名一致
3. wrapper 参数顺序与 C++ adapter 一致
4. 测试中的 reference 计算路径固定（优先 torch baseline）

## Failure Taxonomy

### Compile-time Failures

- 典型原因：声明/定义签名不一致、CMake 未纳入子目录、模板宏缺失
- 修复顺序：签名 -> CMake -> 模板宏

### Import-time Failures

- 典型原因：动态库加载顺序不当、符号未导出、`torch.ops` 未注册
- 修复顺序：`torch_catlass/__init__.py` -> `src/catlass_torch.cpp` -> 导出符号

### Runtime Failures

- 典型原因：`TParams/Params` 填充错误、dtype/layout 映射错误、workspace 参数不一致
- 修复顺序：参数转换 -> shape 推导 -> dtype/layout -> workspace

## Extension Strategy

技能扩展时遵循：

1. 先复用已有统一结构体
2. 只有语义不兼容时才新增结构体
3. 新增结构体时必须给出字段职责说明（避免隐式约定）
4. 每新增 example 都补一条编号接口注释和对应测试

## Deliverable Standard

每次执行技能输出统一四段：

1. 文件变更（按 `include/kernels/src/python/tests` 分组）
2. ABI 变更说明（新增或复用的结构体）
3. 验证结果（build/import/pytest）
4. 遗留风险（仅环境或数据依赖）

