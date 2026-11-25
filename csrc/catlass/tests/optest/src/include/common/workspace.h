/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_WORKSPACE_H
#define OPTEST_WORKSPACE_H

#include <cstddef>

#include <acl/acl.h>

namespace catlass_torch {
namespace common {
/**
 * @brief Allocate temporary memory for certain operator. For being managed by torch_npu, do not use `aclrtMalloc` or
 * `aclrtMallocAlign32` to allocate workspace memory.
 *
 * @param wkspAddr An address pointing to the workspace address variable.
 * @param wkspSize The size of the workspace to allocate.
 * @return aclError
 */
aclError workspaceMalloc(void** wkspAddr, size_t wkspSize);
} // namespace common
} // namespace catlass_torch

#endif
