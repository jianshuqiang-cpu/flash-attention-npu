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

#ifndef OPTEST_JIT_CONFIG_H
#define OPTEST_JIT_CONFIG_H

#include <cstdlib>
#include <string>
#include <vector>

namespace CatlassKernel {

/**
 * @brief Kernel type classification for JIT compilation.
 *
 * Determines the Ascend C kernel execution mode and corresponding compiler flags.
 */
enum JitKernelType : int
{
    AIC = 0, ///< Cube kernel (__cube__)
    AIV = 1, ///< Vector kernel (__vector__)
    MIX = 2  ///< Mixed cube+vector kernel (__mix__)
};

namespace JitConfig {

// ── 环境变量名称 ──
inline constexpr const char* kLogLevelEnv = "CATLASS_JIT_LOG_LEVEL";
inline constexpr const char* kCacheDirEnv = "CATLASS_JIT_CACHE_DIR";
inline constexpr const char* kSanitizeEnv = "MS_SANITIZE_MEMORY";
inline constexpr const char* kVersionEnv = "CATLASS_JIT_VERSION";
inline constexpr const char* kAscendHomeEnv = "ASCEND_HOME_PATH";
inline constexpr const char* kPkgDirEnv = "CATLASS_JIT_PKG_DIR";

inline constexpr const char* kAicAsMix = "CATLASS_JIT_AIC_AS_MIX";
inline constexpr const char* kAivAsMix = "CATLASS_JIT_AIV_AS_MIX";
inline constexpr const char* kMixCV11 = "CATLASS_JIT_MIX_CV_11";

// ── 默认路径 ──
inline constexpr const char* kDefaultCacheDir = "/tmp/catlass_jit";
inline constexpr const char* kHomeCacheSubdir = ".cache/catlass/jit_cache";

// ── 编译器候选路径后缀 ──
inline constexpr const char* kCcecSuffixes[] = {
    "/compiler/bin/ccec",
    "/tools/bisheng_compiler/bin/ccec",
    "/bin/ccec",
};

/**
 * @brief Return compiler flags shared by all JIT kernel compilations.
 */
inline std::vector<std::string> BaseFlags()
{
    return {"-std=c++17", "-O2", "-shared", "-fPIC"};
}

/**
 * @brief Return compiler flags derived from the target NPU architecture.
 * @param arch CATLASS architecture id, for example "2201" or "3510".
 */
inline std::vector<std::string> ArchFlags(const std::string& arch)
{
    return {"--npu-arch=dav-" + arch, "-DCATLASS_ARCH=" + arch};
}

/**
 * @brief Compiler flags for a given kernel type.
 *
 * Maps JitKernelType to the corresponding -DKERNEL_TYPE=... flag.
 * Supports environment variable overrides:
 *   - CATLASS_JIT_AIC_AS_MIX  — emit __mix__(1,0) instead of __cube__ for AIC
 *   - CATLASS_JIT_AIV_AS_MIX  — emit __mix__(0,1) instead of __vector__ for AIV
 *   - CATLASS_JIT_MIX_CV_11   — emit __mix__(1,1) instead of __mix__(1,2) for MIX
 * @param kt Kernel type to generate flags for.
 */
inline std::vector<std::string> KernelTypeFlags(JitKernelType kt)
{
    switch (kt) {
        case AIC: {
            if (std::getenv(kAicAsMix))
                return {"-DKERNEL_TYPE=__mix__(1,0)"};
            return {"-DKERNEL_TYPE=__cube__"};
        }
        case AIV: {
            if (std::getenv(kAivAsMix))
                return {"-DKERNEL_TYPE=__mix__(0,1)"};
            return {"-DKERNEL_TYPE=__vector__"};
        }
        case MIX: {
            if (std::getenv(kMixCV11))
                return {"-DKERNEL_TYPE=__mix__(1,1)"};
            return {"-DKERNEL_TYPE=__mix__(1,2)"};
        }
    }
    return {};
}

/**
 * @brief Build the preprocessor define carrying the package version.
 * @param version Version string exported by the Python package loader.
 */
inline std::string VersionDefine(const std::string& version)
{
    return "-DCATLASS_VERSION_FULL=" + version;
}

} // namespace JitConfig
} // namespace CatlassKernel
#endif // OPTEST_JIT_CONFIG_H