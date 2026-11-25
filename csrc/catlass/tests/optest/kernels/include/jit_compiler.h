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

#ifndef OPTEST_JIT_COMPILER_H
#define OPTEST_JIT_COMPILER_H

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

#include "jit_config.h"
#include "jit_logger.h"
#include "jit_util.h"

namespace CatlassKernel {

/** @brief Compiled JIT kernel ABI entry function pointer type. */
using JitEntryFn = void (*)(uint32_t blockNum, aclrtStream stream, const void* params);

/**
 * @brief RAII wrapper around ``dlopen`` and ``dlclose``.
 */
class SharedLib {
public:
    /** @brief Construct an empty library handle. */
    SharedLib() noexcept = default;

    /**
     * @brief Open a shared library immediately.
     * @param path Path to the compiled shared object.
     * @throws std::runtime_error when ``dlopen`` fails.
     */
    explicit SharedLib(const std::string& path)
    {
        handle_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            const char* err = ::dlerror();
            JIT_THROW(std::string("dlopen failed: ") + (err ? err : "unknown") + ", path=" + path);
        }
    }

    /** @brief Close the library handle when present. */
    ~SharedLib()
    {
        if (handle_)
            ::dlclose(handle_);
    }

    /**
     * @brief Move ownership from another library handle.
     * @param other Source handle that becomes empty.
     */
    SharedLib(SharedLib&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    /**
     * @brief Replace this handle with another handle by move assignment.
     * @param other Source handle that becomes empty.
     * @return Reference to this handle.
     */
    SharedLib& operator=(SharedLib&& other) noexcept
    {
        if (this != &other) {
            if (handle_)
                ::dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    SharedLib(const SharedLib&) = delete;
    SharedLib& operator=(const SharedLib&) = delete;

    /**
     * @brief Resolve a symbol from the loaded library.
     * @param name Symbol name.
     * @return Raw symbol pointer.
     * @throws std::runtime_error when the symbol is absent.
     */
    [[nodiscard]] void* sym(const std::string& name) const
    {
        ::dlerror();
        auto* ptr = ::dlsym(handle_, name.c_str());
        const char* err = ::dlerror();
        if (err || ptr == nullptr) {
            JIT_THROW(std::string("dlsym failed: ") + (err ? err : "null symbol") + ", symbol=" + name);
        }
        return ptr;
    }

    /**
     * @brief Report whether this object currently owns a loaded handle.
     */
    explicit operator bool() const noexcept
    {
        return handle_ != nullptr;
    }

private:
    void* handle_ = nullptr;
};

/**
 * JitCompiler — 单例，管理 JIT 编译 & 缓存
 *
 * 编译器本身不依赖任何 kernel 类型。调用方负责：
 *   1. 用 JitMacroGenerator<TParams>::generate(kernelName, tParams) 构建编译宏
 *   2. 传入 getKernel(templatePath, macros)
 * 入口符号统一约定为 "run"，缓存名从 macros["CATLASS_KERNEL_NAME"] 提取。
 */
class JitCompiler {
public:
    /**
     * @brief Return the process-wide JIT compiler instance.
     */
    static JitCompiler& instance();

    /**
     * @brief Return a compiled kernel entry for a template and macro set.
     *
     * The method first checks the in-memory cache, then a disk cache, and only
     * invokes bisheng when no matching shared object exists.
     *
     * @param templatePath Template path relative to the resolved template base.
     * @param macros Preprocessor macros that define the template specialization.
     * @param kt Kernel type (AIC/AIV/MIX) for compiler flag selection.
     * @return ABI-compatible ``run`` symbol from the loaded shared object.
     */
    JitEntryFn getKernel(const char* templatePath, const MacroMap& macros, JitKernelType kt = AIC);

    /**
     * @brief Drop in-memory loaded kernel handles.
     *
     * Disk cache files are intentionally preserved so future calls can reload
     * already compiled kernels without rebuilding them.
     */
    void clearCache();

private:
    JitCompiler() = default;
    ~JitCompiler();

    void lazyInit();

    /**
     * @brief Compile one template specialization into a shared object.
     * @param name Kernel short name used for diagnostics and fallback macro naming.
     * @param templatePath Path to the Ascend C template source file.
     * @param macros Preprocessor definitions for template specialization.
     * @param kt Kernel type used to select the KERNEL_TYPE compiler flag.
     * @param soPath Output path for the compiled shared object.
     */
    void compile(
        std::string_view name, std::string_view templatePath, const MacroMap& macros, JitKernelType kt,
        const std::string& soPath);

    /**
     * @brief Build the bisheng command line for a JIT compilation.
     * @param name Kernel short name used for diagnostics and fallback macro naming.
     * @param templatePath Path to the Ascend C template source file.
     * @param macros Preprocessor definitions for template specialization.
     * @param kt Kernel type used to select the KERNEL_TYPE compiler flag.
     * @param soPath Output path for the compiled shared object.
     * @return Complete compiler argument vector ready for subprocess invocation.
     */
    [[nodiscard]] std::vector<std::string> buildCompilerArgs(
        std::string_view name, std::string_view templatePath, const MacroMap& macros, JitKernelType kt,
        const std::string& soPath);

    std::once_flag initFlag_;
    std::string cacheDir_;
    std::string bishengPath_;
    std::string npuArch_;
    std::string templateBase_;

    struct LoadedKernel {
        SharedLib lib;
        JitEntryFn entry{nullptr};
    };

    std::unordered_map<std::string, LoadedKernel> loaded_;
    std::mutex mutex_;
};

} // namespace CatlassKernel
#endif // OPTEST_JIT_COMPILER_H
