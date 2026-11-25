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
#include "catlass/gemm/kernel/w4a4_matmul_per_token_per_channel_dequant.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/workspace_alloc.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A int4b_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B int4b_t
#endif
#define CATLASS_JIT_ELEMENT_C float
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B zN
#endif

#ifndef CATLASS_JIT_BLOCK_SCHEDULER
#define CATLASS_JIT_BLOCK_SCHEDULER 30
#endif

using namespace Catlass;

using AscendC::int4b_t;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementScale = uint64_t;
using ElementPerTokenScale = float;
using ElementD = bfloat16_t;

using LayoutA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutScale = layout::VectorLayout;
using LayoutPerTokenScale = layout::VectorLayout;
using LayoutD = layout::RowMajor;

using ArchTag = Arch::AtlasA2;
constexpr uint32_t preloadStages = 1;
constexpr uint32_t l1Stages = 2;
constexpr uint32_t l0AStages = 2;
constexpr uint32_t l0BStages = 2;
constexpr uint32_t l0CStages = 1;
constexpr bool enableUnitFlag = false;
constexpr bool enableShuffleK = true;
using DispatchPolicy = Gemm::MmadAtlasA2W4A4MatmulPerTokenPerChannelDequant<
    preloadStages, l1Stages, l0AStages, l0BStages, l0CStages, enableUnitFlag, enableShuffleK>;

using L1TileShape = GemmShape<128, 256, 1024>;
using L0TileShape = GemmShape<128, 256, 256>;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<half, layout::RowMajor>;
using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;

using TileCopyMmad = Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, void, ScaleGranularity::PER_CHANNEL>;
using BlockMmad =
    Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>;

using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2W4A4PerTokenPerChannelDequant;
using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>;
using DType = Gemm::GemmType<ElementD, LayoutD>;

using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

using EpilogueTileShape = MatrixShape<48, 256>;
using TileBroadcastOneBlk =
    Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
using TileOneBlkColumnBroadcastMul =
    Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
using TileCopy = Epilogue::Tile::TileCopyW4A4Gemm<ArchTag, CType, PerTokenScaleType, DType>;
using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, PerTokenScaleType, DType, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
    TileCopy, TileScheduler>;

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<
    (CATLASS_JIT_BLOCK_SCHEDULER / 10), (CATLASS_JIT_BLOCK_SCHEDULER % 10)>;
constexpr uint32_t workspaceStages = 2;
using MatmulKernel = Gemm::Kernel::W4A4MatmulPerTokenPerChannelDequant<
    BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages, ElementScale, LayoutScale>;

using ProblemShape = Catlass::GemmCoord;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceScale = params->inputAddr[2];
    uint8_t* devicePerTokenScale = params->inputAddr[3];
    uint8_t* deviceD = params->outputAddr[0];

    LayoutA layoutA{m, k};
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutScale layoutScale{n};
    LayoutPerTokenScale layoutPerTokenScale{m};
    LayoutD layoutD{m, n};

    typename MatmulKernel::Arguments arguments{
        ProblemShape{m, n, k}, blockNum,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceScale, layoutScale,
        devicePerTokenScale, layoutPerTokenScale,
        deviceD, layoutD};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
