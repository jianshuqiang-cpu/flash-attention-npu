/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_RUN_NPU_FUNC_H
#define OPTEST_RUN_NPU_FUNC_H

#include <acl/acl.h>

#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

/**
 * @brief Launch a CATLASS kernel through TorchNPU's OpCommand wrapper.
 *
 * The macro checks the function pointer before dispatch and converts C++
 * exceptions raised by the kernel launcher into ACL internal errors so the
 * TorchNPU runtime can surface a consistent failure.
 */
#define RUN_NPU_FUNC(func, ...)                                                                                    \
    do {                                                                                                           \
        if ((func) == nullptr) {                                                                                   \
            throw std::runtime_error(                                                                              \
                std::string("Kernel '") + #func + "' is not available on this NPU architecture. "                  \
                "At " + __FILE__ + ":" + std::to_string(__LINE__));                                                \
        }                                                                                                          \
    } while (false);                                                                                               \
    at_npu::native::OpCommand::RunOpApiV2(#func, [=]() -> aclError {                                               \
        try {                                                                                                      \
            func(__VA_ARGS__);                                                                                     \
        } catch (...) {                                                                                            \
            return ACL_ERROR_INTERNAL_ERROR;                                                                       \
        }                                                                                                          \
        return ACL_SUCCESS;                                                                                        \
    })

#endif
