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

#ifndef OPTEST_BROADCAST_MATMUL_PERBLOCK_QUANT_H
#define OPTEST_BROADCAST_MATMUL_PERBLOCK_QUANT_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

template <auto KernelFunc>
struct BroadcastMatmulPerblockQuantLike {
    using OutputType = std::tuple<at::Tensor, at::Tensor>;

    static void GetKernelInfo(
        const at::Tensor& a, const at::Tensor& b,
        CatlassKernel::MatmulParams& params)
    {
        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(a.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(b.storage().data()));

        uint32_t batchCount = static_cast<uint32_t>(a.size(0));
        uint32_t m = static_cast<uint32_t>(a.size(1));
        uint32_t k = static_cast<uint32_t>(a.size(2));
        uint32_t n = static_cast<uint32_t>(b.size(1));

        params.batch = batchCount;
        params.m = m;
        params.k = k;
        params.n = n;
    }

    static OutputType AllocOutput(
        CatlassKernel::MatmulParams& params, const at::Tensor& refTensor)
    {
        auto dst = at::empty({params.batch, params.m, params.n}, refTensor.options().dtype(torch::kFloat8_e4m3fn));
        auto scale = at::empty({params.batch}, refTensor.options().dtype(torch::kFloat));

        params.outputAddr.resize(2);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(dst.storage().data()));
        params.outputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(scale.storage().data()));

        return {dst, scale};
    }

    static OutputType Run(const at::Tensor& a, const at::Tensor& b)
    {
        CatlassKernel::MatmulParams params;
        GetKernelInfo(a, b, params);
        auto output = AllocOutput(params, a);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
