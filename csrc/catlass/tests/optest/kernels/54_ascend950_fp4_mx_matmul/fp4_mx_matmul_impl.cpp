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

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float4_e2m1x2_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float4_e2m1x2_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementBias = void;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceMxScaleA = params->inputAddr[2];
    uint8_t* deviceMxScaleB = params->inputAddr[3];
    uint8_t* deviceC = params->outputAddr[0];
    uint8_t* deviceBias = nullptr;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, ElementBias>;
    using MxBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Catlass::Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;
    using MatmulKernel = Catlass::Gemm::Kernel::MxMatmulTla<MxBlockMmad, BlockEpilogue, BlockScheduler>;

    typename MatmulKernel::Arguments arguments{
        problemShape, deviceA, layoutA, deviceB, layoutB, deviceMxScaleA, layoutMxScaleA,
        deviceMxScaleB, layoutMxScaleB, deviceC, layoutC, deviceBias};

    uint64_t taskNum64 = static_cast<uint64_t>(CeilDiv(m, tla::get<0>(L1TileShape{}))) *
                         static_cast<uint64_t>(CeilDiv(n, tla::get<1>(L1TileShape{})));
    uint32_t taskNum = static_cast<uint32_t>(std::min(taskNum64, static_cast<uint64_t>(UINT32_MAX)));
    uint32_t aicCoreUsed = std::min(blockNum, taskNum);

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}
