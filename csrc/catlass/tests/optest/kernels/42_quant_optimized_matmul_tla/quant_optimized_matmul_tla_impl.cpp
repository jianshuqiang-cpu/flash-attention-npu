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

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/quant_optimized_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/workspace_alloc.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int8_t
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C int32_t
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B ColumnMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif
#ifndef CATLASS_JIT_ELEMENT_SCALE
#define CATLASS_JIT_ELEMENT_SCALE float
#endif
#ifndef CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE
#define CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE float
#endif
#ifndef CATLASS_JIT_ELEMENT_D
#define CATLASS_JIT_ELEMENT_D bfloat16_t
#endif

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementScale = CATLASS_JIT_ELEMENT_SCALE;
using ElementPerTokenScale = CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE;
using ElementD = CATLASS_JIT_ELEMENT_D;
using ArchTag = Arch::AtlasA2;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::RowMajor;
using LayoutTagScale = layout::RowMajor;
using LayoutTagPerTokenScale = LayoutTagScale;
using LayoutTagD = LayoutTagC;

template <class LayoutTag>
auto GetPaddingLayout(LayoutTag layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<LayoutTag, layout::RowMajor>) {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols), static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols));
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(
            MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows),
                RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)));
        return MakeLayout(shape, stride);
    }
}

const uint32_t align = 256;

template <class LayoutTag>
size_t GetWorkspaceLen(LayoutTag layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows)
         * RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

using L1TileShape = std::conditional_t<
    std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
    Shape<_256, _128, _512>, Shape<_128, _256, _512>>;
using L0TileShape = std::conditional_t<
    std::is_same_v<LayoutTagA, layout::ColumnMajor> && std::is_same_v<LayoutTagB, layout::ColumnMajor>,
    Shape<_256, _128, _128>, Shape<_128, _256, _128>>;

constexpr bool enableUnitFlag = false;
constexpr bool enableShuffleK = true;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

constexpr uint32_t ubStages = 2;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantTla<ubStages>;
using ElementCompute = float;
using EpilogueTileShape = MatrixShape<32, 256>;

using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlkTla<ArchTag, ElementCompute, EpilogueTileShape::ROW>;
using TileOneBlkColumnBroadcastMul =
    Epilogue::Tile::TileOneBlkColumnBroadcastMulTla<ArchTag, ElementCompute, EpilogueTileShape>;
using EpilogueTileCopy = Epilogue::Tile::TileCopyDequantTla<
    ArchTag, ElementC, LayoutTagC, ElementScale, LayoutTagScale, ElementPerTokenScale, LayoutTagPerTokenScale, ElementD,
    LayoutTagD>;
using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, ElementC, ElementScale, ElementPerTokenScale, ElementD, TileRowBroadcastMul,
    TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, EpilogueTileCopy, EpilogueTileScheduler>;

constexpr uint32_t workspaceStages = 2;
#ifndef CATLASS_JIT_NEED_PADDING_A
#define CATLASS_JIT_NEED_PADDING_A false
#endif
#ifndef CATLASS_JIT_NEED_PADDING_B
#define CATLASS_JIT_NEED_PADDING_B false
#endif

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    LayoutTagScale tagScale = LayoutTagScale::MakeLayout<ElementScale>(1, n);
    LayoutTagPerTokenScale tagPerTokenScale = LayoutTagPerTokenScale::MakeLayout<ElementPerTokenScale>(1, m);

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    auto layoutScale = MakeLayoutFromTag(tagScale);
    auto layoutPerTokenScale = MakeLayoutFromTag(tagPerTokenScale);
    auto layoutD = MakeLayoutFromTag(tagC);

    using TensorA = Tensor<
        AscendC::GlobalTensor<ElementA>, decltype(layoutA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorB = Tensor<
        AscendC::GlobalTensor<ElementB>, decltype(layoutB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorC = Tensor<
        AscendC::GlobalTensor<ElementC>, decltype(layoutC), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceScale = params->inputAddr[2];
    uint8_t* devicePerTokenScale = params->inputAddr[3];
    uint8_t* deviceD = params->outputAddr[0];

#if defined(CATLASS_JIT_NEED_PADDING_A) && CATLASS_JIT_NEED_PADDING_A
    auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
    constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
    using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, computeLengthA>;
#else
    auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
    using PaddingA = void;
#endif
#if defined(CATLASS_JIT_NEED_PADDING_B) && CATLASS_JIT_NEED_PADDING_B
    auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
    constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
    using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, computeLengthB>;
#else
    auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
    using PaddingB = void;
#endif
    using TensorWA = Tensor<
        AscendC::GlobalTensor<ElementA>, decltype(layoutWA), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TensorWB = Tensor<
        AscendC::GlobalTensor<ElementB>, decltype(layoutWB), tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
    using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<
        ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB, TensorC, LayoutTagC, void, void,
        CATLASS_JIT_NEED_PADDING_A, CATLASS_JIT_NEED_PADDING_B>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB, TensorC, void, TileCopy>;

    using MatmulKernel = Gemm::Kernel::QuantOptimizedMatmulTla<
        BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB, workspaceStages>;

    uint8_t* deviceWA = deviceA;
    uint8_t* deviceWB = deviceB;
#if CATLASS_JIT_NEED_PADDING_A
    {
        size_t sizeWA = GetWorkspaceLen(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{})) * sizeof(ElementA);
        deviceWA = g_catlassWorkspaceAlloc(sizeWA);
    }
#endif
#if CATLASS_JIT_NEED_PADDING_B
    {
        size_t sizeWB = GetWorkspaceLen(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(ElementB);
        deviceWB = g_catlassWorkspaceAlloc(sizeWB);
    }
#endif

    typename MatmulKernel::Arguments arguments{
        Catlass::GemmCoord{m, n, k},
        blockNum,
        {deviceA, layoutA, deviceB, layoutB, deviceWA, layoutWA, deviceWB, layoutWB},
        {deviceScale, layoutScale, devicePerTokenScale, layoutPerTokenScale, deviceD, layoutD}};
    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
