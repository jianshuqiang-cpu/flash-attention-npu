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

namespace {

uint32_t CeilDiv256(uint32_t v)
{
    return (v + 255U) / 256U;
}

} // namespace

/**
 * @brief example 61_ascend950_svd_quant_matmul: Resolve and launch the JIT SvdQuant matmul implementation.
 */
extern "C" void Ascend950SvdQuantMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const SvdQuantMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_svd_quant_matmul", tParams);
    const uint32_t blocks = CeilDiv256(params.m) * CeilDiv256(params.n);
    macros["CATLASS_JIT_SVD_TILING"] = (blocks < blockNum) ? "1" : "0";
    macros["L2_CACHE_HINT"] = "1";
    auto* entry = JitCompiler::instance().getKernel(
        "svd_quant_matmul_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
