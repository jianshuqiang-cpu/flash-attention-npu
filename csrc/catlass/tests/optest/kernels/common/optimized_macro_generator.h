/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_KERNELS_COMMON_OPTIMIZED_MACRO_GENERATOR_H
#define OPTEST_KERNELS_COMMON_OPTIMIZED_MACRO_GENERATOR_H

#include <cstdint>

#include "catlass/layout/layout.hpp"
#include "common.h"

/**
 * @brief Inject shape-dependent JIT macros for QuantOptimizedMatmulTLA.
 *
 * Computes CATLASS_JIT_NEED_PADDING_A / B (padding flags) and
 * CATLASS_JIT_BLOCK_SCHEDULER (scheduler variant) from raw shape
 * and layout parameters then merges them into `macros`.
 *
 * @param macros   Macro map to extend (in/out).
 * @param m        Rows of A / M-dimension.
 * @param n        Cols of B / N-dimension.
 * @param k        Reduction dimension.
 * @param isNzA    True if A uses blocked (zN/nZ) layout.
 * @param isTransA True if A is column-major.
 * @param isNzB    True if B uses blocked (zN/nZ) layout.
 * @param isTransB True if B is column-major.
 * @param align    Element alignment threshold (default 256).
 */
inline void ApplyOptMacros(
    std::unordered_map<std::string, std::string>& macros,
    uint32_t m, uint32_t n, uint32_t k,
    bool isNzA, bool isTransA,
    bool isNzB, bool isTransB,
    uint32_t align = 256)
{
    auto needPadded = [align](bool isNz, bool isTrans, uint32_t rows, uint32_t cols) -> bool {
        if (isNz) return false;
        if (!isTrans) return CatlassKernel::IsNeedPadding(Catlass::layout::RowMajor(rows, cols), align);
        return CatlassKernel::IsNeedPadding(Catlass::layout::ColumnMajor(rows, cols), align);
    };

    macros["CATLASS_JIT_NEED_PADDING_A"] = needPadded(isNzA, isTransA, m, k) ? "1" : "0";
    macros["CATLASS_JIT_NEED_PADDING_B"] = needPadded(isNzB, isTransB, k, n) ? "1" : "0";
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = (m > n) ? "30" : "31";
}

#endif // OPTEST_KERNELS_COMMON_OPTIMIZED_MACRO_GENERATOR_H
