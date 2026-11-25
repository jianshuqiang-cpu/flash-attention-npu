/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
#include "catlass/gemm/kernel/quant_multi_core_splitk_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
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

using L1TileShape = Shape<_128, _256, _512>;
using L0TileShape = Shape<_128, _256, _128>;

constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

using TileCopy = Gemm::Tile::PackedTileCopyTla<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

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

using ElementAccumulator = typename BlockMmad::ElementAccumulator;
static constexpr uint32_t computeLength = 192 * 1024 / sizeof(ElementAccumulator);
using ReduceAdd = Catlass::Gemm::Kernel::SplitkReduceAdd<ArchTag, ElementAccumulator, ElementC, 1, computeLength>;

using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;

using MatmulKernel = Gemm::Kernel::QuantMultiCoreSplitkMatmulTla<
    BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);
    auto layoutC = MakeLayoutFromTag(tagC);
    LayoutTagScale tagScale = LayoutTagScale::MakeLayout<ElementScale>(1, n);
    auto layoutScale = MakeLayoutFromTag(tagScale);
    LayoutTagPerTokenScale tagPerTokenScale = LayoutTagPerTokenScale::MakeLayout<ElementPerTokenScale>(1, m);
    auto layoutPerTokenScale = MakeLayoutFromTag(tagPerTokenScale);
    auto layoutD = MakeLayoutFromTag(tagC);

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceScale = params->inputAddr[2];
    uint8_t* devicePerTokenScale = params->inputAddr[3];
    uint8_t* deviceD = params->outputAddr[0];

    typename MatmulKernel::Arguments arguments{
        Catlass::GemmCoord{m, n, k},
        blockNum,
        deviceA,
        deviceB,
        deviceScale,
        devicePerTokenScale,
        deviceD,
        layoutA,
        layoutB,
        layoutScale,
        layoutPerTokenScale,
        layoutD};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
