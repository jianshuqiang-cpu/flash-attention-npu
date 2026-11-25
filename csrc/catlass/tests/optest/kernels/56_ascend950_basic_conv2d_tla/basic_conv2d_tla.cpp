/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <acl/acl.h>
#include <iostream>
#include <stdexcept>

#include "catlass/conv/kernel/basic_conv2d_tla.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel_prebuilt.h"
#include "../common/workspace_alloc.h"

namespace CatlassKernel {
using namespace Catlass;

#define ACL_CHECK(status)                                                                   \
    do {                                                                                    \
        aclError error = status;                                                            \
        if (error != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl; \
        }                                                                                   \
    } while (0)

template <class Kernel>
CATLASS_GLOBAL __mix__(1, 0) void BasicConv2dTlaKernelLauncher(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

void Ascend950BasicConv2dTLA(const uint32_t blockNum, aclrtStream stream, const ConvParams& params)
{
    uint32_t batch = params.fmapRelated[0];
    uint32_t hi = params.fmapRelated[1];
    uint32_t wi = params.fmapRelated[2];
    uint32_t cin = params.fmapRelated[3];
    uint32_t cout = params.fmapRelated[4];

    uint8_t kh = params.filterRelated[0];
    uint8_t kw = params.filterRelated[1];

    uint8_t padLeft = params.padList[0];
    uint8_t padRight = params.padList[1];
    uint8_t padTop = params.padList[2];
    uint8_t padBottom = params.padList[3];

    uint8_t strideH = params.strideList[0];
    uint8_t strideW = params.strideList[1];

    uint8_t dilationH = params.dilationList[0];
    uint8_t dilationW = params.dilationList[1];

    Conv2dParams problemParams(
        batch, hi, wi, cin, cout, kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW);

    uint8_t* deviceFmap = params.inputAddr.at(0);
    uint8_t* deviceFilter = params.inputAddr.at(1);
    uint8_t* deviceOutput = params.outputAddr.at(0);

    using ArchTag = Arch::Ascend950;
    constexpr uint32_t L1A_STAGES = 2;
    constexpr uint32_t L1B_STAGES = 2;
    constexpr uint32_t L0A_STAGES = 2;
    constexpr uint32_t L0B_STAGES = 2;
    constexpr uint32_t L0C_STAGES = 1;
    constexpr bool ENABLE_UNIT_FLAG = false;
    using DispatchPolicy =
        Conv::ConvPingpong<ArchTag, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;
    using FmapL1TileShape = Conv2dFmapL1Shape<8, 12, 8>;
    using FilterL1TileShape = Conv2dFilterL1Shape<96, 8>;
    using L0TileShape = Conv2dL0Shape<16, 96, 16>;

    using ElementFmap = half;
    using ElementFilter = half;
    using ElementOutput = half;

    using BlockConv2d = Conv::Block::BlockConv2dTla<
        DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape, ElementFmap, ElementFilter, ElementOutput>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

    using Conv2dKernel = Conv::Kernel::BasicConv2dTla<BlockConv2d, BlockEpilogue, BlockScheduler>;

    typename Conv2dKernel::Arguments arguments{problemParams, deviceFmap, deviceFilter, deviceOutput};
    if (!Conv2dKernel::CanImplement(arguments)) {
        throw std::runtime_error("Conv2d op cannot be implemented: L1TileShape/L0TileShape exceeds the L1 space");
    }
    size_t sizeWorkspace = Conv2dKernel::GetWorkspaceSize(arguments);
    uint8_t* deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        deviceWorkspace = g_catlassWorkspaceAlloc(sizeWorkspace);
    }
    typename Conv2dKernel::Params kernelParams = Conv2dKernel::ToUnderlyingArguments(arguments, deviceWorkspace);

    BasicConv2dTlaKernelLauncher<Conv2dKernel><<<blockNum, nullptr, stream>>>(kernelParams);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}

} // namespace CatlassKernel
