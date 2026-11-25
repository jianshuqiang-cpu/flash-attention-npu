/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/workspace_alloc.h"

extern "C" {

WorkspaceAllocFn g_catlassWorkspaceAlloc = nullptr;
WorkspaceFreeFn  g_catlassWorkspaceFree  = nullptr;
WorkspaceAllocCopyFn g_catlassWorkspaceAllocFromHost = nullptr;

void CatlassSetWorkspaceAlloc(WorkspaceAllocFn alloc)
{
    g_catlassWorkspaceAlloc = alloc;
}

void CatlassSetWorkspaceFree(WorkspaceFreeFn free_fn)
{
    g_catlassWorkspaceFree = free_fn;
}

void CatlassSetWorkspaceAllocFromHost(WorkspaceAllocCopyFn allocCopy)
{
    g_catlassWorkspaceAllocFromHost = allocCopy;
}

} // extern "C"
