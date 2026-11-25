/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

/**
 * @brief example 52_quant_multi_core_splitk_matmul_tla: Resolve and launch the JIT-specialized quantized multi-core splitK matmul implementation.
 */
extern "C" void QuantMultiCoreSplitkMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("quant_multi_core_splitk_matmul_tla", tParams);
    auto* entry = JitCompiler::instance().getKernel("quant_multi_core_splitk_matmul_tla_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
