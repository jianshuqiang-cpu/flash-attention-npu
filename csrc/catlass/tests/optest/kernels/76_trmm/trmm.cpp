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

#include <iostream>

namespace CatlassKernel {

namespace {
constexpr uint32_t SMALL_RIGHT_N_LIMIT = 512u;
constexpr uint32_t SMALL_RIGHT_K128_N_LIMIT = 256u;
constexpr uint32_t SMALL_LEFT_M_LIMIT = 512u;
constexpr uint32_t SMALL_LEFT_K128_M_LIMIT = 256u;

enum class TileVariant : uint32_t
{
    TILE_DEFAULT = 0,
    TILE_SMALL_RIGHT_LOWER = 1,
    TILE_SMALL_RIGHT_UPPER = 2,
    TILE_SMALL_RIGHT_K128 = 3,
    TILE_DEFAULT_SWIZZLE31 = 4,
    TILE_SMALL_LEFT_LOWER = 5,
    TILE_SMALL_LEFT_UPPER = 6,
    TILE_SMALL_LEFT_K128 = 7,
};

TileVariant SelectTileVariant(const TrmmParams& params)
{
    uint32_t effectiveUplo = params.uplo ^ params.trans;
    bool isUpper = (effectiveUplo == 1u);
    bool isLower = (effectiveUplo == 0u);

    if (params.side == 1) {
        if (isUpper && params.n <= SMALL_RIGHT_K128_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_K128;
        if (isUpper && params.n <= SMALL_RIGHT_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_UPPER;
        if (isLower && params.n <= SMALL_RIGHT_N_LIMIT)
            return TileVariant::TILE_SMALL_RIGHT_LOWER;
    } else {
        if (isUpper && params.m <= SMALL_LEFT_K128_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_K128;
        if (isUpper && params.m <= SMALL_LEFT_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_UPPER;
        if (isLower && params.m <= SMALL_LEFT_M_LIMIT)
            return TileVariant::TILE_SMALL_LEFT_LOWER;
    }

    if (params.m >= 2048 && params.m < params.n)
        return TileVariant::TILE_DEFAULT_SWIZZLE31;
    return TileVariant::TILE_DEFAULT;
}

const char* SchedulerMacro(TileVariant variant)
{
    switch (variant) {
        case TileVariant::TILE_SMALL_RIGHT_LOWER:
            return "10";
        case TileVariant::TILE_SMALL_RIGHT_UPPER:
            return "41";
        case TileVariant::TILE_SMALL_RIGHT_K128:
        case TileVariant::TILE_SMALL_LEFT_K128:
            return "11";
        case TileVariant::TILE_DEFAULT_SWIZZLE31:
            return "31";
        case TileVariant::TILE_DEFAULT:
        case TileVariant::TILE_SMALL_LEFT_LOWER:
        case TileVariant::TILE_SMALL_LEFT_UPPER:
        default:
            return "30";
    }
}
} // namespace

/**
 * @brief example 76_trmm: Resolve and launch the JIT-specialized TRMM implementation.
 */
extern "C" void Trmm(const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const TrmmParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("trmm", tParams);
    auto variant = SelectTileVariant(params);
    macros["CATLASS_JIT_TRMM_TILE_VARIANT"] = std::to_string(static_cast<uint32_t>(variant));
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = SchedulerMacro(variant);

    auto* entry = JitCompiler::instance().getKernel("trmm_impl.cpp", macros, JitKernelType::MIX);
    if (entry == nullptr) {
        std::cerr << "[ERROR] Failed to resolve JIT kernel: trmm_impl.cpp" << std::endl;
        return;
    }
    entry(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
