/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_JIT_MACRO_GENERATOR_H
#define OPTEST_JIT_MACRO_GENERATOR_H

#include <string>
#include <unordered_map>
#include <vector>

#include <acl/acl.h>

#include "catlass_kernel.h"

namespace CatlassKernel {

/**
 * @brief Default JIT macro generation policy.
 *
 * 主模板不产生任何宏。各 kernel 类型应通过偏特化实现自己的
 * 宏生成逻辑。
 */
template <typename TParams, typename = void>
struct JitMacroGenerator {
    static void appendTo(std::unordered_map<std::string, std::string>& /*macros*/, const TParams& /*p*/)
    {}

    static std::unordered_map<std::string, std::string> generate(const char* kernelName, const TParams& p)
    {
        std::unordered_map<std::string, std::string> macros;
        macros["CATLASS_KERNEL_NAME"] = kernelName;
        JitMacroGenerator::appendTo(macros, p);
        macros["CATLASS_JIT_KERNEL_NAME"] = makeKernelName(macros, p);
        return macros;
    }

    static std::string makeKernelName(
        const std::unordered_map<std::string, std::string>& /*macros*/, const TParams& /*p*/)
    {
        return {};
    }
};

/**
 * @brief JitMacroGenerator 对 TParams 的特化（实现见 .cpp）。
 *
 * 遍历 element / transpose 两个 map 动态生成宏，不做任何硬编码假设。
 */
template <>
struct JitMacroGenerator<TParams> {
    static void appendTo(std::unordered_map<std::string, std::string>& macros, const TParams& p);
    static std::unordered_map<std::string, std::string> generate(const char* kernelName, const TParams& p);
    static std::string makeKernelName(
        const char* kernelName, const std::unordered_map<std::string, std::string>& macros);
};

} // namespace CatlassKernel

#endif // OPTEST_JIT_MACRO_GENERATOR_H
