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

namespace CatlassKernel {

extern "C" void Ascend950Fp8MxGroupedMatmulFinalizeRouting(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams,
    const GroupedMxFinalizeRoutingParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_fp8_mx_grouped_matmul_finalize_routing", tParams);
    macros["CATLASS_JIT_ENABLE_BIAS"] = (params.inputAddr.size() > 7 && params.inputAddr[7] != nullptr) ? "true" : "false";
    macros["CATLASS_JIT_ENABLE_SHARED_INPUT"] = (params.inputAddr.size() > 8 && params.inputAddr[8] != nullptr) ? "true" : "false";
    auto* entry = JitCompiler::instance().getKernel(
        "fp8_mx_grouped_matmul_finalize_routing_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
