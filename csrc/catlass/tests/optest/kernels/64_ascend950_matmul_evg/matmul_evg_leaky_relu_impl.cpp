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
#include "catlass/gemm/kernel/basic_matmul_tla_visitor.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

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

using BaseL1 = tuple<C<256>, C<256>, C<128>>;
using BaseL0 = tuple<C<256>, C<256>, C<32>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

#ifndef CATLASS_JIT_LEAKY_RELU_SLOPE
#define CATLASS_JIT_LEAKY_RELU_SLOPE 0.1f
#endif

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulEvgParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    constexpr uint32_t evgUbNodes = 2;    // AccLoad + Compute；Store 不占
    constexpr uint32_t evgUbStages = 2;   // epilogue 双缓冲
    constexpr uint32_t computeLength = RoundDown(
        ArchTag::UB_SIZE / evgUbNodes / evgUbStages / sizeof(ElementC), BYTE_PER_C0); // 每槽元素上限，向下取 BYTE_PER_C0 整数倍
    using LayoutC = decltype(layoutC);
    using EVG = Epilogue::Fusion::TreeVisitor<
        Epilogue::Fusion::VisitorAuxStore<ElementC, LayoutC>,
        Epilogue::Fusion::TreeVisitor<
            Epilogue::Fusion::VisitorCompute<Epilogue::Fusion::LeakyRelu, ElementC, ElementC>,
            Epilogue::Fusion::VisitorAccLoad<ElementC>>>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueVisitor<>, ArchTag, Int<computeLength>, EVG, ElementC>;
    using MatmulKernel = Gemm::Kernel::BasicMatmulTlaVisitor<BlockMmad, BlockEpilogue, BlockScheduler>;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceD = params->outputAddr[0];

    const float negativeSlope = CATLASS_JIT_LEAKY_RELU_SLOPE;

    typename EVG::Arguments evg_args{
        {{}, {{negativeSlope}}},
        {deviceD, layoutC}};

    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k}, deviceA, layoutA, deviceB, layoutB, nullptr, layoutC, nullptr, evg_args};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
