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

#include "jit_compiler.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>

#include <unistd.h>

#include "jit_config.h"
#include "jit_sha256.h"

namespace fs = std::filesystem;

namespace CatlassKernel {

JitCompiler& JitCompiler::instance()
{
    static JitCompiler inst;
    return inst;
}

JitCompiler::~JitCompiler()
{
    clearCache();
}

void JitCompiler::lazyInit()
{
    std::call_once(initFlag_, [this] {
        std::error_code ec;

        const char* cacheDirEnv = std::getenv(JitConfig::kCacheDirEnv);
        std::string cacheDir = (cacheDirEnv && *cacheDirEnv) ? cacheDirEnv : "";
        if (cacheDir.empty()) {
            const char* home = std::getenv("HOME");
            if (home)
                cacheDir = std::string(home) + "/" + JitConfig::kHomeCacheSubdir;
        }
        if (cacheDir.empty()) {
            cacheDir = JitConfig::kDefaultCacheDir;
        }
        fs::create_directories(cacheDir, ec);
        JIT_CHECK(!ec, "mkdir failed: " + cacheDir + ": " + ec.message());

        const char* ver = std::getenv(JitConfig::kVersionEnv);
        if (ver && *ver) {
            cacheDir += "/" + std::string(ver);
            fs::create_directories(cacheDir, ec);
            JIT_CHECK(!ec, "mkdir failed: " + cacheDir + ": " + ec.message());
        }

        cacheDir_ = std::move(cacheDir);

        bishengPath_ = FindCompilerPath();
        JIT_CHECK(!bishengPath_.empty(), "bisheng compiler not found (set ASCEND_HOME_PATH)");

        npuArch_ = GetCurrentNPUArch();

        templateBase_ = ResolveTemplateBase();

        JIT_LOG(
            JitLogLevel::Info, "JIT init: cache=%s compiler=%s arch=%s template=%s", cacheDir_.c_str(),
            bishengPath_.c_str(), npuArch_.c_str(), templateBase_.empty() ? "(none)" : templateBase_.c_str());
    });
}

JitEntryFn JitCompiler::getKernel(const char* templatePath, const MacroMap& macros, JitKernelType kt)
{
    lazyInit();

    const auto nameIt = macros.find("CATLASS_KERNEL_NAME");
    JIT_CHECK(nameIt != macros.end() && !nameIt->second.empty(), "CATLASS_KERNEL_NAME not set in macros");
    const std::string& targetName = nameIt->second;
    const std::string entrySymbol = "run";

    JIT_CHECK(templatePath && templatePath[0], "templatePath is empty");

    auto makeKernelUuid = [&](const MacroMap& macroValues) -> std::string {
        std::vector<std::pair<std::string, std::string>> sorted;
        sorted.reserve(macroValues.size() + 2);
        for (const auto& kv : macroValues) {
            sorted.emplace_back(kv.first, kv.second);
        }
        sorted.emplace_back("__ARCH__", npuArch_);
        sorted.emplace_back("__KT__", std::to_string(static_cast<int>(kt)));
        std::sort(sorted.begin(), sorted.end());

        std::string input;
        for (const auto& kv : sorted) {
            input += kv.first + "=" + kv.second + "&";
        }
        return Sha256::hash(input);
    };

    const std::string cacheKey = makeKernelUuid(macros);
    const std::string soPath = cacheDir_ + "/" + cacheKey + ".so";

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = loaded_.find(cacheKey);
        if (it != loaded_.end()) {
            JIT_LOG(JitLogLevel::Debug, "mem hit: %s %s", cacheKey.c_str(), targetName.c_str());
            return it->second.entry;
        }
    }

    {
        std::error_code ec;
        if (!fs::is_regular_file(soPath, ec)) {
            std::lock_guard<std::mutex> lk(mutex_);

            {
                auto it = loaded_.find(cacheKey);
                if (it != loaded_.end())
                    return it->second.entry;
            }

            if (!fs::is_regular_file(soPath, ec)) {
                JIT_LOG(JitLogLevel::Info, "compiling: %s \xe2\x86\x92 %s", targetName.c_str(), soPath.c_str());

                fs::create_directories(fs::path(soPath).parent_path(), ec);
                JIT_CHECK(!ec, "mkdir failed: " + std::string(fs::path(soPath).parent_path()) + ": " + ec.message());
                compile(targetName, templatePath, macros, kt, soPath);
            }
        }
    }

    SharedLib lib(soPath);
    auto* entry = reinterpret_cast<JitEntryFn>(lib.sym(entrySymbol));

    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = loaded_.find(cacheKey);
        if (it != loaded_.end()) {
            return it->second.entry;
        }
        loaded_.emplace(cacheKey, LoadedKernel{std::move(lib), entry});
    }

    return entry;
}

void JitCompiler::compile(
    std::string_view name, std::string_view templatePath, const MacroMap& macros, JitKernelType kt,
    const std::string& soPath)
{
    auto args = buildCompilerArgs(name, templatePath, macros, kt, soPath);

    auto cmdJoin = [](const std::vector<std::string>& args) -> std::string {
        std::string cmd;
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0)
                cmd += ' ';
            cmd += args[i];
        }
        return cmd;
    };

    const std::string cmdStr = cmdJoin(args);
    JIT_LOG(JitLogLevel::Debug, "compile: %s", cmdStr.c_str());

    const ProcessResult result = RunProcessCapture(args);
    if (result.exitCode != 0) {
        (void)::unlink(soPath.c_str());
        JIT_LOGE("compile failed (exit=%d): %s", result.exitCode, result.output.c_str());
        JIT_THROW(
            "compile failed (exit=" + std::to_string(result.exitCode) + ")\n" + "command: " + cmdStr + "\n" +
            "output:\n" + result.output);
    }

    {
        std::error_code ec;
        JIT_CHECK(fs::is_regular_file(soPath, ec), "compiler succeeded but output not created: " + soPath);
    }
}

std::vector<std::string> JitCompiler::buildCompilerArgs(
    std::string_view name, std::string_view templatePath, const MacroMap& macros, JitKernelType kt,
    const std::string& soPath)
{
    std::vector<std::string> args;
    args.reserve(32);

    args.push_back(bishengPath_);

    args.push_back("-x");
    args.push_back("asc");

    {
        auto base = JitConfig::BaseFlags();
        args.insert(args.end(), base.begin(), base.end());
    }

    {
        auto archFlags = JitConfig::ArchFlags(npuArch_);
        args.insert(args.end(), archFlags.begin(), archFlags.end());
    }

    if (macros.count("KERNEL_TYPE")) {
        JIT_LOG(JitLogLevel::Info, "KERNEL_TYPE already set in macros, skip KernelTypeFlags");
    } else {
        auto ktFlags = JitConfig::KernelTypeFlags(kt);
        args.insert(args.end(), ktFlags.begin(), ktFlags.end());
    }

    {
        const char* ms = std::getenv(JitConfig::kSanitizeEnv);
        if (ms && std::string(ms) == "1") {
            args.push_back("-g");
            args.push_back("--cce-enable-sanitizer");
            JIT_LOG(JitLogLevel::Info, "msSanitizer ENABLED (MS_SANITIZE_MEMORY=1)");
        }
    }

    {
        const char* ver = std::getenv(JitConfig::kVersionEnv);
        args.push_back(JitConfig::VersionDefine(ver ? ver : "unknown"));
    }

    auto sortedMacros = [](const MacroMap& macros) {
        std::vector<std::pair<std::string, std::string>> result;
        result.reserve(macros.size());
        for (const auto& kv : macros) {
            JIT_CHECK(!kv.first.empty(), "empty macro name");
            result.emplace_back(kv.first, kv.second);
        }
        std::sort(result.begin(), result.end());
        return result;
    };

    for (const auto& kv : sortedMacros(macros)) {
        args.push_back("-D" + kv.first + "=" + kv.second);
    }

    {
        auto it = macros.find("CATLASS_JIT_KERNEL_NAME");
        std::string kn = (it != macros.end() && !it->second.empty()) ? it->second : std::string(name);
        args.push_back("-DKERNEL_NAME=" + kn);
        kn += "_arch" + npuArch_;
        args.push_back("-DCATLASS_JIT_KERNEL_NAME=" + kn);
    }

    auto extraIncludes = BuildIncludeArgsFromEnv();
    args.insert(args.end(), extraIncludes.begin(), extraIncludes.end());

    args.push_back(templateBase_.empty() ? std::string(templatePath) : templateBase_ + std::string(templatePath));
    args.push_back("-o");
    args.push_back(soPath);

    return args;
}

void JitCompiler::clearCache()
{
    std::lock_guard<std::mutex> lk(mutex_);
    loaded_.clear();
}

} // namespace CatlassKernel
