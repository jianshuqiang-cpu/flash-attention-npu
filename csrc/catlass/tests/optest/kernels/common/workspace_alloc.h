/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Workspace 分配器注入接口。
 *
 * 全局变量 g_catlassWorkspaceAlloc 定义在 libcatlass_kernel_jit_compiler.so 中
 * （以 RTLD_GLOBAL 加载），JIT 模板 .so 和 torch 层通过同一符号共享。
 */

#ifndef OPTEST_KERNELS_COMMON_WORKSPACE_ALLOC_H
#define OPTEST_KERNELS_COMMON_WORKSPACE_ALLOC_H

#include <cstddef>
#include <cstdint>

using WorkspaceAllocFn     = uint8_t* (*)(size_t size);
using WorkspaceFreeFn      = void (*)(uint8_t* ptr, size_t size);
using WorkspaceAllocCopyFn = uint8_t* (*)(const void* hostData, size_t size);

extern "C" {

extern WorkspaceAllocFn     g_catlassWorkspaceAlloc;
extern WorkspaceFreeFn      g_catlassWorkspaceFree;
extern WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost;

void CatlassSetWorkspaceAlloc(WorkspaceAllocFn alloc);
void CatlassSetWorkspaceFree(WorkspaceFreeFn free_fn);
void CatlassSetWorkspaceAllocFromHost(WorkspaceAllocCopyFn allocCopy);

} // extern "C"

#endif // OPTEST_KERNELS_COMMON_WORKSPACE_ALLOC_H
