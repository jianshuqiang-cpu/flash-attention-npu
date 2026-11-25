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

extern "C" void W4A4MatmulPerTokenPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("w4a4_matmul_per_token_per_channel_dequant", tParams);
    macros["CATLASS_JIT_ELEMENT_A"] = "int4b_t";
    macros["CATLASS_JIT_ELEMENT_B"] = "int4b_t";
    macros["CATLASS_JIT_ELEMENT_C"] = "half";
    macros["CATLASS_JIT_KERNEL_NAME"] =
        JitMacroGenerator<TParams>::makeKernelName("w4a4_matmul_per_token_per_channel_dequant", macros);
    auto* entry = JitCompiler::instance().getKernel(
        "w4a4_matmul_per_token_per_channel_dequant_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
