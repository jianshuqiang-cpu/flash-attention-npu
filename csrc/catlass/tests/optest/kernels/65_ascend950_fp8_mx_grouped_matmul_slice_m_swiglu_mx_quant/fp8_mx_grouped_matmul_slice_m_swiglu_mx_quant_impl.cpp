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

#include <acl/acl.h>
#include <iostream>
#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif
#ifndef CATLASS_JIT_ELEMENT_Q
#define CATLASS_JIT_ELEMENT_Q float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_Q_SCALE
#define CATLASS_JIT_ELEMENT_Q_SCALE float8_e8m0_t
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
using ElementC = float;
using ElementQ = CATLASS_JIT_ELEMENT_Q;
using ElementQScale = CATLASS_JIT_ELEMENT_Q_SCALE;
using ElementGluRes = bfloat16_t;
using ElementGroupList = int64_t;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagQ = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using MmadDispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
using EpilogueDispatchPolicy = Catlass::Epilogue::BlockEpilogueSwigluMxQuant;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<128>>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::GroupedMatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t N = params->n;
    uint32_t k = params->k;
    uint32_t groupCount = params->batch;
    uint32_t N_half = N / 2;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceMxScaleA = params->inputAddr[2];
    uint8_t* deviceMxScaleB = params->inputAddr[3];
    uint8_t* deviceGroupList = params->inputAddr[4];
    uint8_t* deviceQ = params->outputAddr[0];
    uint8_t* deviceQScale = params->outputAddr[1];

    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    Catlass::GemmCoord problemShape{m, N, k};

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, N);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, N);
    auto layoutQ = tla::MakeLayout<ElementQ, LayoutTagQ>(m, N_half);
    auto layoutQScale = tla::MakeMxScaleLayout<ElementQScale, LayoutTagQ, false>(
        m, CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(N_half));

    using TileCopy = Catlass::Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
        ElementMxScale, decltype(layoutMxScaleA),
        ElementMxScale, decltype(layoutMxScaleB),
        ElementC, LayoutTagQ, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, L0TileShape, ElementC, ElementC, ElementGluRes, ElementQ, ElementQScale>;
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    using MatmulKernel = Catlass::Gemm::Kernel::GroupedMxMatmulSliceMSwigluMxQuantTla<
        BlockMmad, BlockEpilogue, BlockScheduler, ElementGroupList, decltype(layoutQ), decltype(layoutQScale)>;

    typename BlockEpilogue::Params epilogueParams;
    epilogueParams.baseM = m;
    epilogueParams.baseN = N_half;
    epilogueParams.baseK = k;

    typename MatmulKernel::Arguments arguments{
        problemShape, groupCount,
        deviceGroupList,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceMxScaleA, layoutMxScaleA,
        deviceMxScaleB, layoutMxScaleB,
        deviceQ, layoutQ,
        deviceQScale, layoutQScale,
        epilogueParams
    };

    std::vector<int64_t> hostGroupList(groupCount);
    ACL_CHECK(aclrtMemcpy(hostGroupList.data(), groupCount * sizeof(int64_t),
                deviceGroupList, groupCount * sizeof(int64_t),
                ACL_MEMCPY_DEVICE_TO_HOST));

    constexpr int64_t L1_TILE_M = 128;
    constexpr int64_t L1_TILE_N = 256;
    int64_t totalCoreLoops = 0;
    for (int64_t g = 0; g < groupCount; ++g) {
        int64_t currentM = static_cast<int64_t>(hostGroupList[g]);
        if (currentM == 0) continue;
        int64_t loopsM = (currentM + L1_TILE_M - 1) / L1_TILE_M;
        int64_t loopsN = (N_half + L1_TILE_N - 1) / L1_TILE_N;
        totalCoreLoops += loopsM * loopsN;
        if (totalCoreLoops >= blockNum) {
            totalCoreLoops = blockNum;
            break;
        }
    }

    Catlass::RunKernel<MatmulKernel>(arguments, stream, totalCoreLoops);
}
