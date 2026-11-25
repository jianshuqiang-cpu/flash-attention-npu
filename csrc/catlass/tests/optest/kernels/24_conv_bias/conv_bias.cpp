/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include <stdexcept>

#include "catlass_kernel_prebuilt.h"
#include "../common/workspace_alloc.h"

namespace CatlassKernel {

template <class Kernel>
CATLASS_GLOBAL __mix__(1, 0) void ConvBiasKernelLauncher(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

void ConvBias(const uint32_t blockNum, aclrtStream stream, const ConvParams& params)
{
    using namespace Catlass;

    using ElementFmap = half;
    using ElementFilter = half;
    using ElementBias = half;
    using ElementOut = half;
    using ArchTag = Arch::AtlasA2;

    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 1;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;

    using DispatchPolicy =
        Conv::ConvAtlasA2Pingpong<l1AStages, l1BStages, l0AStages, l0BStages, l0CStages, enableUnitFlag>;

    using LayoutFmap = layout::NDC1HWC0;
    using LayoutFilter = layout::KDC1KHKWN1N0C0;
    using LayoutOut = layout::NDC1HWC0;
    using LayoutBias = layout::VectorLayout;

    using FmapType = Gemm::GemmType<ElementFmap, LayoutFmap>;
    using FilterType = Gemm::GemmType<ElementFilter, LayoutFilter>;
    using BiasType = Gemm::GemmType<ElementBias, LayoutBias>;
    using OutType = Gemm::GemmType<ElementOut, LayoutOut>;

    using CoreTileShape = ConvCoreShape<2, 2, 2, 2>;
    using FmapL1TileShape = ConvFmapL1Shape<16, 1, 1>;
    using FilterL1TileShape = ConvFilterL1Shape<1, 1, 16>;
    using L0TileShape = ConvL0Shape<16, 16, 16>;

    using BlockConv = Conv::Block::BlockConv<
        DispatchPolicy, CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutType,
        BiasType>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    using ConvKernel = Conv::Kernel::ConvBias<BlockConv, BlockEpilogue, BlockScheduler>;

    Conv3dParams problemShape = Conv3dParams::MakeConvCoord(
        params.fmapRelated.data(), params.filterRelated.data(), params.padList.data(), params.strideList.data(),
        params.dilationList.data());

    typename ConvKernel::Arguments arguments{
        problemShape, params.inputAddr[0], params.inputAddr[1], params.outputAddr[0], params.inputAddr[2]};

    if (!ConvKernel::CanImplement(arguments)) {
        throw std::runtime_error("Conv2d op cannot be implemented: L1TileShape/L0TileShape exceeds the L1 space");
    }
    size_t sizeWorkspace = ConvKernel::GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        deviceWorkspace = g_catlassWorkspaceAlloc(sizeWorkspace);
    }
    typename ConvKernel::Params kernelParams = ConvKernel::ToUnderlyingArguments(arguments, deviceWorkspace);

    ConvBiasKernelLauncher<ConvKernel><<<blockNum, nullptr, stream>>>(kernelParams);
}

} // namespace CatlassKernel
