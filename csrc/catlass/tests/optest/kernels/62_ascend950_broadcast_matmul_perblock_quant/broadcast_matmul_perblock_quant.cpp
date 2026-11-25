/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_perblock_quant.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/broadcast_matmul_perblock_quant_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "common/common.h"
#include "common/kernel_runner.h"
#include "catlass_kernel.h"

using namespace Catlass;
using namespace tla;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;
using ElementDst = float8_e4m3_t;
using ElementScale = float;

using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::RowMajor;
using LayoutTagC = layout::RowMajor;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
constexpr uint32_t l0CStages = 1;
constexpr bool enableL1Resident = true;

using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32, l0CStages, enableL1Resident>;
using L1TileShape = Shape<Int<128>, Int<128>, Int<128>>;
using L0TileShape = Shape<Int<128>, Int<128>, Int<128>>;

using TileCopy =
    Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

using TilePerBlockQuant = Epilogue::Tile::TilePerBlockQuant<ArchTag, ElementC, ElementDst, ElementScale>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    Epilogue::EpilogueAscend950PerBlockQuantTla<1>,
    ElementC,
    ElementDst,
    ElementScale,
    TilePerBlockQuant
>;

using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

using MatmulKernel = Gemm::Kernel::BroadcastMatmulPerblockQuantTla<BlockMmad, BlockEpilogue, BlockScheduler>;

static void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    auto* deviceA = params->inputAddr[0];
    auto* deviceB = params->inputAddr[1];
    auto* deviceDst = params->outputAddr[0];
    auto* deviceScale = params->outputAddr[1];

    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t batchCount = params->batch;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

    GemmCoord problemShape(m, n, k);
    typename MatmulKernel::Arguments arguments{
        batchCount, problemShape, deviceA, layoutA, deviceB, layoutB, layoutC, deviceDst, deviceScale};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}

namespace CatlassKernel {

void BroadcastMatmulPerblockQuant(
    const uint32_t blockNum, aclrtStream stream, const MatmulParams& params)
{
    run(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
