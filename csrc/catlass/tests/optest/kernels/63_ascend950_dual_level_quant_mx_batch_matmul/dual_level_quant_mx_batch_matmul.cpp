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

extern "C" void Ascend950DualLevelQuantMxBatchMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate(
        "ascend950_dual_level_quant_mx_batch_matmul", tParams);
    macros["L2_CACHE_HINT"] = "1";

    const uint32_t minMn = params.m < params.n ? params.m : params.n;
    const uint32_t maxMn = params.m < params.n ? params.n : params.m;
    if (params.batch == 1 && minMn >= 12000 && maxMn >= 65000 && params.k >= 7000) {
        macros["CATLASS_EXAMPLE63_USE_PRELOAD"] = "1";
    }

    auto* entry = JitCompiler::instance().getKernel(
        "dual_level_quant_mx_batch_matmul_impl.cpp",
        macros,
        JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
