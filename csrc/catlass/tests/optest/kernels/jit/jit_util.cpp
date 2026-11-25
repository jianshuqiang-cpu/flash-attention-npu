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

#include "jit_util.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <sys/wait.h>
#include <tiling/platform/platform_ascendc.h>

#include "jit_config.h"

#ifndef ENABLE_ASCEND950
#define ENABLE_ASCEND950 1
#endif

namespace fs = std::filesystem;

namespace CatlassKernel {

namespace {

/**
 * @brief Read a non-empty environment variable or return an empty C string.
 */
const char* EnvOrEmpty(const char* name)
{
    const char* v = std::getenv(name);
    return (v && *v) ? v : "";
}

} // anonymous namespace

namespace {

std::string shellQuote(const std::string& arg)
{
    std::string result = "'";
    for (char c : arg) {
        if (c == '\'') {
            result += "'\\''";
        } else {
            result += c;
        }
    }
    result += "'";
    return result;
}

} // anonymous namespace

ProcessResult RunProcessCapture(const std::vector<std::string>& args)
{
    JIT_THROW_IF(args.empty() || args[0].empty(), "empty compiler command");

    std::string cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0)
            cmd += ' ';
        cmd += shellQuote(args[i]);
    }
    cmd += " 2>&1";

    FILE* fp = ::popen(cmd.c_str(), "r");
    JIT_THROW_IF(!fp, std::string("popen failed: ") + std::strerror(errno));

    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        output += buf;
    }

    const int status = ::pclose(fp);

    ProcessResult result;
    result.output = std::move(output);
    if (WIFEXITED(status)) {
        result.exitCode = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exitCode = 128 + WTERMSIG(status);
    } else {
        result.exitCode = -1;
    }
    return result;
}

std::string GetCurrentNPUArch()
{
    const platform_ascendc::SocVersion socVersion =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetSocVersion();
    switch (socVersion) {
        case platform_ascendc::SocVersion::ASCEND910B:
        case platform_ascendc::SocVersion::ASCEND910_93:
            return "2201";
#if ENABLE_ASCEND950
        case platform_ascendc::SocVersion::ASCEND950:
            return "3510";
#endif
        default:
            JIT_THROW("unsupported Ascend SocVersion: " + std::to_string(static_cast<int>(socVersion)));
    }
}

std::string FindCompilerPath()
{
    std::error_code ec;
    const char* ascendHome = EnvOrEmpty(JitConfig::kAscendHomeEnv);
    if (*ascendHome) {
        for (const char* suffix : JitConfig::kCcecSuffixes) {
            const std::string path = std::string(ascendHome) + suffix;
            if (fs::is_regular_file(path, ec)) {
                return path;
            }
        }
    }

    return "ccec";
}

std::vector<std::string> BuildIncludeArgsFromEnv()
{
    std::vector<std::string> args;

    const char* pkgDir = EnvOrEmpty(JitConfig::kPkgDirEnv);
    if (*pkgDir) {
        args.push_back("-I" + std::string(pkgDir) + "/include");
        args.push_back("-I" + std::string(pkgDir) + "/jit");
    }

    return args;
}

std::string ResolveTemplateBase()
{
    const char* pkgDir = EnvOrEmpty(JitConfig::kPkgDirEnv);
    if (*pkgDir) {
        return std::string(pkgDir) + "/jit/templates/";
    }

    return {};
}

} // namespace CatlassKernel
