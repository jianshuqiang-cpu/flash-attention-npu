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

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"
#include "jit_macros.h"

#include <string>
#include <unordered_map>

namespace CatlassKernel {

namespace {

struct EvgDispatch {
    const char* macroName;
    const char* templateFile;
};

const std::unordered_map<std::string, EvgDispatch>& GetEvgDispatchTable()
{
    static const std::unordered_map<std::string, EvgDispatch> table = {
        {"add", {"ascend950_matmul_evg_add", "matmul_evg_add_impl.cpp"}},
        {"add_ub", {"ascend950_matmul_evg_add_ub", "matmul_evg_add_ub_impl.cpp"}},
        {"bias", {"ascend950_matmul_evg_bias", "matmul_evg_bias_impl.cpp"}},
        {"leaky_relu", {"ascend950_matmul_evg_leaky_relu", "matmul_evg_leaky_relu_impl.cpp"}},
        {"sigmoid", {"ascend950_matmul_evg_sigmoid", "matmul_evg_sigmoid_impl.cpp"}},
        {"silu", {"ascend950_matmul_evg_silu", "matmul_evg_silu_impl.cpp"}},
        {"tanh", {"ascend950_matmul_evg_tanh", "matmul_evg_tanh_impl.cpp"}},
    };
    return table;
}

} // namespace

/**
 * @brief example 64_ascend950_matmul_evg: Unified JIT EVG matmul entry.
 */
extern "C" void MatmulEvg(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulEvgParams& params)
{
    const auto& table = GetEvgDispatchTable();
    auto it = table.find(params.evgType);
    JIT_CHECK(it != table.end(), "unsupported evgType: " + params.evgType);

    const auto& dispatch = it->second;
    auto macros = JitMacroGenerator<TParams>::generate(dispatch.macroName, tParams);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = (params.m > params.n) ? "30" : "31";
    if (params.evgType == "leaky_relu") {
        macros["CATLASS_JIT_LEAKY_RELU_SLOPE"] = std::to_string(params.negativeSlope) + "f";
    }

    auto* entry = JitCompiler::instance().getKernel(dispatch.templateFile, macros, JitKernelType::MIX);
    JIT_CHECK(entry != nullptr, std::string("JIT load failed for ") + dispatch.templateFile);
    entry(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
