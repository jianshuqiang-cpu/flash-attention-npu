# Workspace & Memory Allocation

JIT 模板中的 device 内存分配统一使用 `common/workspace_alloc.h` 提供的函数。

## API

| 函数 | 用途 | H2D 拷贝 |
|------|------|----------|
| `g_catlassWorkspaceAlloc(size)` | 分配 device 内存，kernel 直接写 | ❌ |
| `g_catlassWorkspaceAllocFromHost(data, size)` | 分配 device 内存 + 从 host 拷贝数据 | ✅ |

```cpp
#include "common/workspace_alloc.h"

// workspace buffer — kernel 直接写，无需 H2D
uint8_t* gmWorkspace = g_catlassWorkspaceAlloc(m * n * sizeof(float));

// struct 数组 — 需要 H2D：host 填好数据，拷到 device
GemmCoord* dShapes = (GemmCoord*)g_catlassWorkspaceAllocFromHost(
    hostShapes.data(), G * sizeof(GemmCoord));
```

## 规则

1. **禁止 `aclrtMalloc` + `aclrtMemcpy(H2D)`** — torch tensor 内存不支持 `aclrtMemcpy(H2D)`，会 segfault。`g_catlassWorkspaceAllocFromHost` 内部通过 `torch.copy_()` 安全完成 H2D 拷贝。

2. **禁止 `memset` / `aclrtMemset`** — device 内存不能直接 memset。kernel 内部初始化。

3. **无需手动 free** — 分配的内存由 `g_wsPool`（torch tensor 静态池）管理，kernel 执行期间自动保活，无需也禁止手动 `aclrtFree`。

4. **kernel 写入的 buffer 用 `g_catlassWorkspaceAlloc`**（gmWorkspace、deviceWA、deviceWB 等）

5. **需要 H2D 的 struct 数组用 `g_catlassWorkspaceAllocFromHost`**（problem shapes、layouts、alpha/beta 等）

## 实现

`g_catlassWorkspaceAlloc` 和 `g_catlassWorkspaceAllocFromHost` 定义在 `libcatlass_kernel_jit_compiler.so` 中（`workspace_alloc.cpp`），通过 `dlsym` 从 torch 层注入实现（`src/catlass_torch.cpp`）：

- `wsAlloc` → `at::empty({size}, device('npu'))` → 返回 `storage().data()`
- `wsAllocFromHost` → `at::empty({size}, device('npu'))` + `dst.copy_(at::from_blob(src, {size}))`
- 返回的 tensor 存入 `g_wsPool` 保活
