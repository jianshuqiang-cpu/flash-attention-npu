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

#include "catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_no_deter_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle_grouped_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float8_e4m3_t
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

#ifndef CATLASS_JIT_ENABLE_BIAS
#define CATLASS_JIT_ENABLE_BIAS true
#endif
#ifndef CATLASS_JIT_ENABLE_SHARED_INPUT
#define CATLASS_JIT_ENABLE_SHARED_INPUT true
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementC = float;
using ElementBias = std::conditional_t<CATLASS_JIT_ENABLE_BIAS, bfloat16_t, void>;
using ElementSharedInput = std::conditional_t<CATLASS_JIT_ENABLE_SHARED_INPUT, bfloat16_t, void>;
using ElementGroupList = int64_t;
using ElementLogit = float;
using ElementRowIndex = int64_t;
using ElementOut = float;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* baseParams)
{
    auto* params = static_cast<const CatlassKernel::GroupedMxFinalizeRoutingParams*>(baseParams);

    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t problemCount = params->problemCount;
    uint32_t groupListType = params->groupListType;
    float sharedInputWeight = params->sharedInputWeight;
    uint32_t sharedInputOffset = params->sharedInputOffset;
    uint32_t bsdp = params->bsdp;
    uint32_t batch = params->batch;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    uint32_t aicCoreUsed = blockNum;

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceMxScaleA = params->inputAddr[2];
    uint8_t* deviceMxScaleB = params->inputAddr[3];
    uint8_t* deviceGroupList = params->inputAddr[4];
    uint8_t* deviceLogit = params->inputAddr[5];
    uint8_t* deviceRowIndex = params->inputAddr[6];
    uint8_t* deviceBias = params->inputAddr[7];
    uint8_t* deviceSharedInput = params->inputAddr[8];
    uint8_t* deviceOut = params->outputAddr[0];

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using vecTileShape = Catlass::MatrixShape<tla::get<0>(L1TileShape{}) / 2, tla::get<1>(L1TileShape{})>;

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, ElementBias, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadMxFinalizeRoutingTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy>;

    constexpr uint32_t UB_STAGES = 1;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAscend950FinalizeRouting<UB_STAGES>;
    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogueFinalizeRoutingNoDeter<
        EpilogueDispatchPolicy, ArchTag, vecTileShape, ElementC, ElementRowIndex, ElementSharedInput>;

    using BlockScheduler = typename Catlass::Gemm::Block::GemmGroupedAswtTailSplitSwizzle<>;
    using MatmulKernel = Catlass::Gemm::Kernel::GroupedMxMatmulFinalizeRoutingNoDeterTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList, ElementSharedInput>;

    typename MatmulKernel::Arguments arguments{
        aicCoreUsed,
        problemShape,
        problemCount,
        deviceGroupList,
        deviceA,
        layoutA,
        deviceB,
        layoutB,
        deviceMxScaleA,
        layoutMxScaleA,
        deviceMxScaleB,
        layoutMxScaleB,
        nullptr,
        layoutC,
        deviceBias,
        deviceLogit,
        deviceRowIndex,
        deviceSharedInput,
        groupListType,
        sharedInputWeight,
        static_cast<int64_t>(sharedInputOffset),
        static_cast<int64_t>(batch),
        static_cast<int64_t>(bsdp),
        deviceOut};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}
