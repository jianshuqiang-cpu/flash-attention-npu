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

#include "catlass/gemm/kernel/svd_quant_matmul_tla.hpp"

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

#ifndef CATLASS_JIT_SVD_TILING
#define CATLASS_JIT_SVD_TILING 0
#endif

using InDType = half;
using OutDType = half;

using ElementX = InDType;
using ElementSvd1 = InDType;
using ElementSvd2 = InDType;
using ElementSmoothScale = InDType;
using ElementBias = void;
using ElementW = float4_e2m1x2_t;
using ElementMxScale = float8_e8m0_t;
using ElementC1 = InDType;
using ElementQuantX = float4_e2m1x2_t;
using ElementY = OutDType;

using LayoutTagX = Catlass::layout::RowMajor;
using LayoutTagSvd1 = Catlass::layout::ColumnMajor;
using LayoutTagSvd2 = Catlass::layout::ColumnMajor;
using LayoutTagW = Catlass::layout::ColumnMajor;
using LayoutTagSmoothScale = Catlass::layout::RowMajor;
using LayoutTagC1 = Catlass::layout::RowMajor;
using LayoutTagQuantX = LayoutTagX;
using LayoutTagY = Catlass::layout::RowMajor;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;

#if CATLASS_JIT_SVD_TILING == 1
using L1TileShape1 = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<256>>;
using L0TileShape1 = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
using L1TileShape2 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<128>>;
using L0TileShape2 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using L1TileShape3 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<512>>;
using L0TileShape3 = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
#else
using L1TileShape1 = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<256>>;
using L0TileShape1 = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>;
using L1TileShape2 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
using L0TileShape2 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<64>>;
using L1TileShape3 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>;
using L0TileShape3 = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
#endif

using DispatchPolicy1 = Catlass::Gemm::MmadSvd1<ArchTag, enableUnitFlag>;
using TileCopy1 = Catlass::Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementX, LayoutTagX, ElementSvd1, LayoutTagSvd1, ElementC1, LayoutTagC1, void>;
using BlockMmad1 = Catlass::Gemm::Block::BlockMmadTla<
    DispatchPolicy1, L1TileShape1, L0TileShape1, ElementX, ElementSvd1, ElementC1, void, TileCopy1>;

using SmoothQuant = Catlass::Gemm::Kernel::SmoothQuant<
    ArchTag, ElementX, ElementSmoothScale, ElementQuantX, ElementMxScale, LayoutTagX, L1TileShape1>;

using DispatchPolicy2 = Catlass::Gemm::MmadSvd2<ArchTag, enableUnitFlag>;
using TileCopy2 = Catlass::Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementC1, LayoutTagC1, ElementSvd2, LayoutTagSvd2, ElementY, LayoutTagY, ElementBias>;
using BlockMmad2 = Catlass::Gemm::Block::BlockMmadTla<
    DispatchPolicy2, L1TileShape2, L0TileShape2, ElementC1, ElementSvd2, ElementY, ElementBias, TileCopy2>;

static constexpr uint32_t l1ScaleFactorK = 8;
using DispatchPolicy3 = Catlass::Gemm::MmadSvd3<ArchTag, enableUnitFlag, l1ScaleFactorK>;

using BlockEpilogue = void;
using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::SvdQuantMatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t r = params->r;
    float qmax = params->qmax;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    Catlass::GemmCoord problemShape{m, n, k};

    uint8_t* deviceX = params->inputAddr[0];
    uint8_t* deviceSvd1 = params->inputAddr[1];
    uint8_t* deviceSvd2 = params->inputAddr[2];
    uint8_t* deviceW = params->inputAddr[3];
    uint8_t* deviceMxScaleW = params->inputAddr[4];
    uint8_t* deviceSmoothScale = params->inputAddr[5];
    uint8_t* deviceBias = nullptr;
    uint8_t* deviceY = params->outputAddr[0];

    auto layoutX = tla::MakeLayout<ElementX, LayoutTagX>(m, k);
    auto layoutSvd1 = tla::MakeLayout<ElementSvd1, LayoutTagSvd1>(k, r);
    auto layoutSvd2 = tla::MakeLayout<ElementSvd2, LayoutTagSvd2>(r, n);
    auto layoutW = tla::MakeLayout<ElementW, LayoutTagW>(k, n);
    auto layoutMxScaleW = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagW, true>(mxScaleK, n);
    auto layoutSmoothScale = tla::MakeLayout<ElementSmoothScale, LayoutTagSmoothScale>(static_cast<uint32_t>(1), k);
    auto layoutY = tla::MakeLayout<ElementY, LayoutTagY>(m, n);
    auto layoutMxScaleX = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagQuantX, false>(m, mxScaleK);

    using TileCopy3 = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementQuantX, LayoutTagQuantX, ElementW, LayoutTagW, ElementMxScale, decltype(layoutMxScaleX),
        ElementMxScale, decltype(layoutMxScaleW), ElementY, LayoutTagY, void>;
    using BlockMmad3 = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy3, L1TileShape3, L0TileShape3, ElementQuantX, ElementW, ElementY, void, TileCopy3>;

    using MatmulKernel = Catlass::Gemm::Kernel::SvdQuantMatmulTla<
        SmoothQuant, BlockMmad1, BlockMmad2, BlockMmad3, BlockEpilogue, BlockScheduler>;

    typename MatmulKernel::Arguments arguments{
        problemShape, r, qmax,
        deviceX, layoutX, deviceSvd1, layoutSvd1, deviceSvd2, layoutSvd2, deviceW, layoutW, deviceMxScaleW,
        layoutMxScaleW, deviceSmoothScale, layoutSmoothScale, deviceBias, deviceY, layoutY};

    uint32_t taskNum1 = CeilDiv(m, tla::get<0>(L1TileShape1{})) * CeilDiv(r, tla::get<1>(L1TileShape1{}));
    uint32_t taskNum2 = CeilDiv(m, tla::get<0>(L1TileShape3{})) * CeilDiv(n, tla::get<1>(L1TileShape3{}));
    uint32_t aicCoreUsed = std::min(blockNum, std::max(taskNum1, taskNum2));

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}
