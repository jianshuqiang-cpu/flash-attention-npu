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
#ifndef GROUPED_MATMUL_SLICE_M_GELU_LAUNCHER_HPP
#define GROUPED_MATMUL_SLICE_M_GELU_LAUNCHER_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_gelu.hpp"
#include "acl/acl.h"
#include "helper.hpp"
#include "golden.hpp"

using namespace Catlass;
using namespace tla;

template <class LayoutA, class LayoutB>
void grouped_matmul_slice_m_gelu_launcher(
    aclrtStream stream, GemmCoord problemShape, uint32_t problemCount, uint8_t* deviceGroupList, uint8_t* deviceA,
    LayoutA layoutA, uint8_t* deviceB, LayoutB layoutB, uint8_t* deviceO, uint8_t** deviceWorkspaceOut)
{
    using ArchTag = Arch::Ascend950;

    using ElementA = half;
    using LayoutTagA = layout::RowMajor;

    using ElementB = half;
    using LayoutTagB = layout::RowMajor;

    using ElementO = half;
    using LayoutTagO = layout::RowMajor;

    using ElementMmOut = float32_t;
    using LayoutTagMmOut = layout::RowMajor;

    static constexpr uint32_t PRELOAD_STAGES = 1;
    static constexpr uint32_t L1A_STAGES = 2;
    static constexpr uint32_t L1B_STAGES = 2;
    static constexpr uint32_t L0A_STAGES = 2;
    static constexpr uint32_t L0B_STAGES = 2;
    static constexpr uint32_t L0C_STAGES = 1;

    static constexpr bool ENABLE_UNIT_FLAG = true;
    static constexpr bool ENABLE_SHUFFLE_K = false;
    static constexpr bool USE_HF32_MODE = false;
    static constexpr bool ENABLE_L1_RESIDENT = true;

    using DispatchPolicy = Gemm::MmadPreloadAsyncWithCallbackL0CToUB<
        ArchTag, PRELOAD_STAGES, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG,
        ENABLE_SHUFFLE_K, USE_HF32_MODE, ENABLE_L1_RESIDENT>;

    using L1_TILE_M = Int<240>;
    using L1_TILE_N = Int<256>;

    using L1TileShape = Shape<L1_TILE_M, L1_TILE_N, Int<128>>;
    using L0TileShape = Shape<L1_TILE_M, L1_TILE_N, Int<64>>;

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMmOut, LayoutTagMmOut, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false, Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using BlockMmadTla = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementMmOut, void, TileCopy>;

    using UBTIleCopy = Shape<L1_TILE_M, L1_TILE_N>;
    using MmOutType = Gemm::GemmType<ElementMmOut, LayoutTagMmOut, AscendC::TPosition::VECCALC>;
    using OType = Gemm::GemmType<ElementO, LayoutTagO>;
    using EpilogueDispatchPolicy = Epilogue::EpilogueElemWiseNoSourceFromUB;
    using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseGeluRegBase<ArchTag, ElementO, ElementMmOut>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, MmOutType, OType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, MmOutType, OType, TileElemWiseEpilogue, EpilogueTileCopy, UBTIleCopy>;

    uint8_t* deviceWorkspace{nullptr};
    auto aiCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t m = problemShape.m();
    uint32_t n = problemShape.n();

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMGelu<BlockMmadTla, BlockEpilogue, BlockScheduler, int64_t>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulKernel::Arguments arguments{problemShape, problemCount, deviceGroupList, deviceA,
                                      layoutA,      deviceB,      layoutB,         deviceO};

    MatmulAdapter matmul_op;
    if (matmul_op.CanImplement(arguments) != Status::kSuccess) {
        return;
    }
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aiCoreNum);

    *deviceWorkspaceOut = deviceWorkspace;
}
#endif // GROUPED_MATMUL_SLICE_M_GELU_LAUNCHER_HPP
