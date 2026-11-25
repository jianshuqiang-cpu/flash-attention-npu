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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/fusion/fusion.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla_ub_visitor.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using MmadDispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag>;

// Align tile shapes with example 64_ascend950_matmul_evg_add_ub (RowMajor L0C->UB).
using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;

template <class BlockScheduler>
void LaunchAddUbKernel(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulEvgParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    // fp32 output only (optest/API); fixed SPLIT_M matches example 64 add_ub.
    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    constexpr uint32_t evgUbNodes = 2;    // AuxLoad + Compute；AccLoad(UB) / Store 不占
    constexpr uint32_t evgUbStages = 2;   // epilogue 双缓冲
    constexpr uint32_t evgUbBudget = ArchTag::UB_SIZE - ArchTag::L0C_SIZE / 2; // [0, L0C_SIZE/2) 存 MMAD 结果 C，EVG 仅用后半段 UB
    constexpr uint32_t computeLength = RoundDown(
        evgUbBudget / evgUbNodes / evgUbStages / sizeof(ElementC), BYTE_PER_C0); // 每槽元素上限，向下取 BYTE_PER_C0 整数倍
    using LayoutC = decltype(layoutC);
    using EpilogueDispatchPolicy = Epilogue::EpilogueVisitor<true>;
    using EVG = Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>,
        Epilogue::Fusion::TreeVisitor<
            Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::Add, ElementC>,
            Epilogue::Fusion::VisitorAccLoad<ElementC, EpilogueDispatchPolicy::USE_UB_WORKSPACE>,
            Epilogue::Fusion::VisitorAuxLoad<ElementC, LayoutC>>>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, ArchTag, Int<computeLength>, EVG, ElementC>;
    using MatmulKernel = Gemm::Kernel::BasicMatmulTlaUbVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceX = params->inputAddr[2];
    uint8_t* deviceD = params->outputAddr[0];

    typename EVG::Arguments evg_args{
        {{}, {deviceX, layoutC}, {}},
        {deviceD, layoutC}};

    // UB workspace path does not consume a GM C buffer; align with example 64 add_ub.
    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k}, deviceA, layoutA, deviceB, layoutB, nullptr, LayoutC{}, nullptr, evg_args};

    uint32_t taskNum = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = std::min(blockNum, taskNum);
    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulEvgParams* params)
{
    // Match example 64_ascend950_matmul_evg_add_ub swizzle direction selection.
    if (params->m > params->n) {
        LaunchAddUbKernel<Gemm::Block::GemmIdentityBlockSwizzle<3, 0>>(blockNum, stream, params);
    } else {
        LaunchAddUbKernel<Gemm::Block::GemmIdentityBlockSwizzle<3, 1>>(blockNum, stream, params);
    }
}
