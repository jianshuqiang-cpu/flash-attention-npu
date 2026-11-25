---
name: "catlass-example-to-torch-intf"
description: "Migrate catlass examples to PyTorch extension interfaces. Invoke when user wants to create torch-compatible NPU ops from catlass examples."
---

# Catlass Example to Torch Interface Migration Skill

Migrate standalone catlass example code into a PyTorch extension library loadable via `torch.ops.load_library()`.

## Target Project Structure

All migrated code goes into the `catlass_op/` project, adding new files to the existing structure:

```
catlass_op/
├── CMakeLists.txt
├── include/
│   ├── catlass.h
│   └── catlass_kernel.h
├── src/
│   └── catlass_torch.cpp
├── kernel/
│   └── catlass_kernel.asc
└── test_<func_name>.py
```

## Interface Naming

The final Python interface follows the form `torch.ops.catlass.<func_name>`, where `<func_name>` is derived from the example directory name with any trailing numbering removed.

Examples:
- Example `w8a16_matmul/` → `torch.ops.catlass.w8a16_matmul`
- Example `basic_matmul/` → `torch.ops.catlass.basic_matmul`
- Example `quant_matmul_2/` → `torch.ops.catlass.quant_matmul`

In C++, this maps to:
- `TORCH_LIBRARY(catlass, m) { m.def("<func_name>(...) -> ..."); }` — controls `torch.ops.catlass.<func_name>`
- `TORCH_LIBRARY_IMPL(catlass, PrivateUse1, m) { m.impl("<func_name>", catlass_torch::<func_name>); }`
- The C++ function lives in `namespace catlass_torch { ... }` to distinguish from `catlass_kernel`

## Mandatory Rules

### Rule 1: Strictly Copy Include List from Source Example

The `.asc` kernel file MUST use the exact same catlass include list as the source example. Do NOT add, remove, or guess include paths.

**Procedure**:
1. Read the source example's `.cpp` file
2. Extract ALL `#include "catlass/..."` lines (skip example-specific ones like `"golden.hpp"`, `"helper.hpp"`, `"options.hpp"`)
3. Copy them verbatim into the `.asc` file

### Rule 2: Cast to `uint8_t*`, Never to `GM_ADDR`

The ASC compiler does NOT allow `reinterpret_cast` from `void*` to `GM_ADDR` (`__gm__ uint8_t*`).

Always use `reinterpret_cast<uint8_t*>` when converting `void*` to device pointers. `MatmulKernel::Arguments` accepts `uint8_t*` (which is `GM_ADDR` in host context), and `ToUnderlyingArguments` handles the conversion to `Params` with proper `GM_ADDR` fields.

### Rule 3: Use `void*` Instead of `aclrtStream`

`aclrtStream` is `typedef void *aclrtStream`. To avoid ASC compiler type resolution issues, always use `void*` directly in `.asc` files. In `.cpp` files (host compiler), `aclrtStream` can be used normally.

### Rule 4: No Hardware Sync Handling Required

Hardware sync address is NOT needed. Do NOT add any hardware sync-related code (`aclrtGetHardwareSyncAddr`, `SetSyncBaseAddr`, `hardwareSyncAddr`) anywhere in the project — neither in `.asc` files nor in `.cpp` files.

### Rule 5: Workspace Allocation via `GetWorkspaceSize` + `at::empty`

Many kernels require a workspace buffer for intermediate results (e.g., split-K reduction data). Follow this pattern:

1. **Declare** a `get_<func_name>_workspace_size` function in `include/catlass_kernel.h` and `kernel/catlass_kernel.asc`
2. **Implement** it in `kernel/catlass_kernel.asc` by calling `<Kernel>::GetWorkspaceSize(args)` — this is the canonical way to query workspace size from the catlass kernel
3. **Call** it from `src/catlass_torch.cpp` to get the exact size needed
4. **Allocate** using `at::empty(workspaceSize, at::dtype(at::kByte).device(tensor.device()))` — this is the ONLY allowed method for NPU tensor allocation
5. **Pass** `workspace.data_ptr()` to the kernel launch

**⚠️ Critical: Parameter Consistency.** The `get_workspace_size` call and the actual kernel launch **MUST pass identical parameters** — same shapes, same transpose flags, same quantization params, same group sizes, etc. Any parameter that affects the kernel's internal buffer layout or reduction strategy may change `GetWorkspaceSize`'s return value.

Therefore:
- The `get_<func_name>_workspace_size` declaration MUST accept **all** parameters that the kernel's `Arguments` struct uses (beyond just M/N/K)
- The call to `get_workspace_size` in `catlass_torch.cpp` MUST use the **exact same** values as the subsequent kernel launch
- When reading the kernel header, examine `GetWorkspaceSize`'s implementation to see which `Arguments` fields affect workspace size. Do NOT assume M/N/K are sufficient.

If uncertain whether a parameter affects workspace size, **ask the user** rather than guessing.

Example pattern with transposes (workspace size depends only on logical M/N/K, but transa/transb affect how M/N/K are derived):

```cpp
// In include/catlass_kernel.h:
size_t get_<func_name>_workspace_size(uint32_t coreNum, uint32_t M, uint32_t N, uint32_t K);

// In kernel/catlass_kernel.asc:
namespace catlass_kernel {
size_t get_<func_name>_workspace_size(uint32_t coreNum, uint32_t M, uint32_t N, uint32_t K)
{
    GemmCoord problemShape(M, N, K);
    Kernel::Arguments args{problemShape, coreNum, sizeof(float), nullptr, nullptr, nullptr};
    return Kernel::GetWorkspaceSize(args);
}
}

// In src/catlass_torch.cpp:
size_t workspaceSize = catlass_kernel::get_<func_name>_workspace_size(coreNum, M, N, K);
auto workspace = at::empty(workspaceSize, at::dtype(at::kByte).device(aContig.device()));
// ... pass workspace.data_ptr() to kernel ...
```

## Kernel Launch Pattern

Do NOT use `DeviceGemm` or `KernelAdapter`. Write custom `__global__` entry functions and launch them directly.

The pattern (learned from `KernelAdapter`'s implementation):

1. **Host wrapper** constructs `Arguments` → `ToUnderlyingArguments` → `Params`, then launches `__global__` entry
2. **`__global__` entry** receives `Params` and calls `op(params)`

```cpp
__global__ __aicore__ void <func_name>_impl(MatmulKernel::Params params)
{
    MatmulKernel op;
    op(params);
}

namespace catlass_kernel {
void <func_name>(uint32_t coreNum, void* stream, ...)
{
    // ... construct Arguments with uint8_t* pointers (Rule 2) ...
    MatmulKernel::Params params = MatmulKernel::ToUnderlyingArguments(arguments, workspace);
    <func_name>_impl<<<coreNum, nullptr, stream>>>(params);
}
}
```

## Migration Steps

### Step 1: Analyze Source Example

1. Read the example `.cpp` file
2. **Read the kernel header**: Always read `catlass/include/catlass/gemm/kernel/<kernel_type>.hpp` to understand:
   - `Arguments` struct fields
   - `Params` struct fields
   - `ToUnderlyingArguments` signature
   - `GetWorkspaceSize` signature (if workspace is needed)
3. **Special Logic**: Derive from the actual example code — workspace formulas, scheduler branching, prologue handling, etc. Do NOT pre-assume these.

### Step 2: Add Code to catlass_op

Append new functions to the existing `catlass_op/` files:

- **`kernel/catlass_kernel.asc`**: Add type aliases, `__global__` entry functions, and host wrapper function in `catlass_kernel` namespace
- **`include/catlass_kernel.h`**: Add function declarations in `catlass_kernel` namespace
- **`include/catlass.h`**: Add torch interface declarations in `catlass_torch` namespace
- **`src/catlass_torch.cpp`**: Add torch interface implementation and register in existing `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` blocks
- **`test_<func_name>.py`**: Create new test script

### Step 3: Build and Test

```bash
cd catlass_op && mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd .. && python test_<func_name>.py
```

**Always verify build and test pass before delivering. Fix any errors.**

## Common Patterns

### Data Type Mapping

| Torch dtype | C++ type | Catlass type |
|-------------|----------|--------------|
| `torch.float16` | `at::kHalf` | `half` / `fp16_t` |
| `torch.bfloat16` | `at::kBFloat16` | `bfloat16_t` |
| `torch.float32` | `at::kFloat` | `float` |
| `torch.int8` | `at::kChar` | `int8_t` |
| `torch.int32` | `at::kInt` | `int32_t` |

### Device Check

```cpp
TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "Tensor must be on NPU");
```

### RUN_NPU_FUNC Macro

Already defined in `catlass_op/src/catlass_torch.cpp`. Reuse it.

### Workspace Allocation

```cpp
size_t workspaceSize = catlass_kernel::get_<func_name>_workspace_size(coreNum, M, N, K);
auto workspace = at::empty(workspaceSize, at::dtype(at::kByte).device(tensor.device()));
```

## Execution Instructions

1. **Ask for the example path**: The user must specify the exact path to the catlass example to migrate.
2. **Analyze the example**: Read source files AND kernel header. Derive all special logic from actual code, not from pre-baked assumptions.
3. **Read existing catlass_op files**: Always read current `catlass_op/` files to understand existing code and append correctly.
4. **Add code to catlass_op**: Append new functions to existing files following the rules above.
5. **Build and test**: Verify compilation and correctness. Fix any errors.
6. **Test multiple shapes**: Verify with different M/N/K combinations.
