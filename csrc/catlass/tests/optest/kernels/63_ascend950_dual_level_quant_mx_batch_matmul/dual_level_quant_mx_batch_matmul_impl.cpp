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

#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_dual_level_quant_mx.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy_dual_level_quant_mx.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_mx_preload_tla.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_INPUT
#define CATLASS_JIT_ELEMENT_INPUT float16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C bfloat16_t
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif

using ElementInput = CATLASS_JIT_ELEMENT_INPUT;
// Example 63 only emits E2M1 packed FP4 after dual-level quantization.
using ElementA = float4_e2m1x2_t;
using ElementB = float4_e2m1x2_t;
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagInputA = Catlass::layout::RowMajor;
using LayoutTagInputB = Catlass::layout::RowMajor;
using LayoutTagPhysicalB = Catlass::layout::RowMajor;
using LayoutTagA = Catlass::layout::RowMajor;
using LayoutTagB = Catlass::layout::ColumnMajor;
using LayoutTagC = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
#if defined(CATLASS_EXAMPLE63_USE_PRELOAD)
using DispatchPolicy = Catlass::Gemm::MmadMxPreload<ArchTag, 1, enableUnitFlag, 16>;
#else
using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag, 16>;
#endif
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t batchCount = params->batch;
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceInputA = params->inputAddr[0];
    uint8_t* deviceInputB = params->inputAddr[1];
    uint8_t* deviceC = params->outputAddr[0];
    uint8_t* deviceScaleA1 = params->outputAddr[1];
    uint8_t* deviceScaleA2 = params->outputAddr[2];
    uint8_t* deviceScaleB1 = params->outputAddr[3];
    uint8_t* deviceScaleB2 = params->outputAddr[4];
    uint8_t* deviceWorkspace = params->outputAddr[5];

    uint32_t scaleA1K = CeilDiv<512>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    auto layoutInputA = LayoutTagInputA::MakeLayout<ElementInput>(m, k);
    auto layoutInputB = LayoutTagInputB::MakeLayout<ElementInput>(n, k);
    auto layoutOutputA = LayoutTagA::MakeLayout<ElementA>(m, k);
    auto layoutOutputB = LayoutTagPhysicalB::MakeLayout<ElementB>(n, k);
    auto layoutScaleA1 = LayoutTagA::MakeLayout<float>(m, scaleA1K);
    auto layoutScaleA2 = LayoutTagA::MakeLayout<ElementMxScale>(m, mxScaleAlignedK);
    auto layoutScaleB1 = LayoutTagPhysicalB::MakeLayout<float>(n, scaleA1K);
    auto layoutScaleB2 = LayoutTagPhysicalB::MakeLayout<ElementMxScale>(n, mxScaleAlignedK);
    auto layoutQuantA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutQuantB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    using TileCopyMmad = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
#if defined(CATLASS_EXAMPLE63_USE_PRELOAD)
    using BlockMmad = Catlass::Gemm::Block::BlockMmadMxPreloadTla<
        DispatchPolicy, L1TileShape, L0TileShape,
        ElementA, ElementB, ElementC, void, TileCopyMmad>;
#else
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape,
        ElementA, ElementB, ElementC, void, TileCopyMmad>;
#endif

    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAscend950DualLevelQuantMx<ubStages>;
    using QuantSubTileShape = Catlass::MatrixShape<128, 512>;

    using InputType = Catlass::Gemm::GemmType<ElementInput, Catlass::layout::RowMajor>;
    using OutputType = Catlass::Gemm::GemmType<ElementA, Catlass::layout::RowMajor>;
    using Scale1Type = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using Scale2Type = Catlass::Gemm::GemmType<ElementMxScale, Catlass::layout::RowMajor>;

    using TileCopyQuant = Catlass::Epilogue::Tile::TileCopyDualLevelQuantMx<
        ArchTag, InputType, OutputType, Scale1Type, Scale2Type>;
    using BlockQuant = Catlass::Epilogue::Block::BlockQuantDualLevelMx<
        EpilogueDispatchPolicy, QuantSubTileShape, InputType, OutputType, Scale1Type, Scale2Type, TileCopyQuant>;

    if (m > n) {
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using Kernel = Catlass::Gemm::Kernel::DualLevelQuantMxBatchedMatmulTla<
            BlockMmad, BlockQuant, BlockScheduler, ElementInput>;
        typename Kernel::Arguments arguments{
            batchCount, problemShape, deviceInputA, layoutInputA, deviceInputB, layoutInputB, deviceC, layoutC,
            deviceScaleA1, deviceScaleA2, layoutMxScaleA, deviceScaleB1, deviceScaleB2, layoutMxScaleB, layoutOutputA,
            layoutOutputB, layoutScaleA1, layoutScaleA2, layoutScaleB1, layoutScaleB2, layoutQuantA, layoutQuantB};
        auto kernelParams = Kernel::ToUnderlyingArguments(arguments, deviceWorkspace);
        Catlass::KERNEL_NAME<Kernel><<<blockNum, nullptr, stream>>>(kernelParams);
    } else {
        using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using Kernel = Catlass::Gemm::Kernel::DualLevelQuantMxBatchedMatmulTla<
            BlockMmad, BlockQuant, BlockScheduler, ElementInput>;
        typename Kernel::Arguments arguments{
            batchCount, problemShape, deviceInputA, layoutInputA, deviceInputB, layoutInputB, deviceC, layoutC,
            deviceScaleA1, deviceScaleA2, layoutMxScaleA, deviceScaleB1, deviceScaleB2, layoutMxScaleB, layoutOutputA,
            layoutOutputB, layoutScaleA1, layoutScaleA2, layoutScaleB1, layoutScaleB2, layoutQuantA, layoutQuantB};
        auto kernelParams = Kernel::ToUnderlyingArguments(arguments, deviceWorkspace);
        Catlass::KERNEL_NAME<Kernel><<<blockNum, nullptr, stream>>>(kernelParams);
    }
}
