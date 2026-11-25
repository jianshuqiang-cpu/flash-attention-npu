/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_mix_fixpipe_opti.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C float
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

using namespace Catlass;
using namespace tla;

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementBias = void;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

using BaseL1 = tuple<C<256>, C<256>, C<128>>;
using BaseL0 = tuple<C<256>, C<256>, C<64>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BaseL0>::type;

using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
constexpr bool splitM = std::is_same_v<ElementC, ElementAccumulator>;
constexpr auto copyMode = splitM ?
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M :
    Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;

using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, copyMode>;
using TileMmad = Gemm::Tile::TileMmadTla<DispatchPolicy::ArchTag, ElementA, TileCopy::LayoutTagL1A>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy, TileMmad>;
using EpilogueDispatchPolicy = Epilogue::EpilogueAscend950Fixpipe<splitM>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, L0TileShape, ElementC, ElementC>;

using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape>;

using MatmulKernel = Gemm::Kernel::KernelMatmulMixFixpipeOpti<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k},
        params->inputAddr[0], params->inputAddr[1],
        params->outputAddr[0]
    };

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
