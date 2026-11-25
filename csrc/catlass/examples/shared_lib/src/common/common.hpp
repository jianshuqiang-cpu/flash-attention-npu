/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_COMMON_COMMON_HPP
#define SHARED_LIB_COMMON_COMMON_HPP

#include <iostream>

#include <acl/acl.h>

#include "catlass/layout/layout.hpp"

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                   \
    do {                                                                                    \
        aclError error = status;                                                            \
        if (error != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl; \
        }                                                                                   \
    } while (0)

namespace CatlassKernel {
using namespace Catlass;

template <class Adapter>
void RunAdapter(Adapter matmulOp, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum)
{
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(args);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(args, deviceWorkspace);
    matmulOp(stream, aicCoreNum);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

inline bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::zN layout, uint32_t align)
{
    return false;
}

inline bool IsNeedPadding(layout::nZ layout, uint32_t align)
{
    return false;
}

} // namespace CatlassKernel

#endif
