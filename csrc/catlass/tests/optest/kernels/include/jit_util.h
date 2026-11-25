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

#ifndef OPTEST_JIT_UTIL_H
#define OPTEST_JIT_UTIL_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <acl/acl.h>

#include "jit_macros.h"

namespace CatlassKernel {

using MacroMap = std::unordered_map<std::string, std::string>;

/** @brief Exit code and combined stdout/stderr from a child process. */
struct ProcessResult {
    int exitCode{-1};
    std::string output;
};

/**
 * @brief Run a command line and capture stdout/stderr.
 * @param args Command and arguments in execution order.
 * @return Process exit status and captured output.
 */
[[nodiscard]] ProcessResult RunProcessCapture(const std::vector<std::string>& args);

/**
 * @brief Detect the current NPU architecture id from AscendC platform APIs.
 * @return Architecture id accepted by bisheng, for example "2201".
 */
[[nodiscard]] std::string GetCurrentNPUArch();

/**
 * @brief Resolve the bisheng/ccec compiler executable.
 * @return Compiler path from ASCEND_HOME_PATH, or fallback name resolved by PATH.
 */
[[nodiscard]] std::string FindCompilerPath();

/**
 * @brief Build include flags needed by runtime-compiled JIT templates.
 * @return List of ``-I`` compiler arguments.
 */
[[nodiscard]] std::vector<std::string> BuildIncludeArgsFromEnv();

/**
 * @brief Resolve the base directory that contains installed JIT templates.
 * @return Directory prefix ending with '/', or an empty string for source paths.
 */
[[nodiscard]] std::string ResolveTemplateBase();

} // namespace CatlassKernel
#endif // OPTEST_JIT_UTIL_H
