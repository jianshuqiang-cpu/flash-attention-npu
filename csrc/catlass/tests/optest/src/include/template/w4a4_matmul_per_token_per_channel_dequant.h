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

#ifndef OPTEST_W4A4_MATMUL_PER_TOKEN_PER_CHANNEL_DEQUANT_H
#define OPTEST_W4A4_MATMUL_PER_TOKEN_PER_CHANNEL_DEQUANT_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <KernelFn KernelFunc>
struct W4A4MatmulPerTokenPerChannelDequantLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& scale, const at::Tensor& perTokenScale,
        const c10::ScalarType& outDType, bool transA, bool transB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        // A and B are stored as int8 but interpreted as int4b_t by the kernel.
        // Use ACL_INT4 for element mapping.
        tParams.element["A"] = ACL_INT8;
        tParams.element["B"] = ACL_INT8;
        tParams.element["C"] = ACL_FLOAT16;
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = true;  // B uses zN/nZ layout
        tParams.useNz["C"] = false;

        int64_t kPhysA = mat1.size(1);  // physical K for A: K/2
        int64_t kPhysB = mat2.size(0);  // physical K for B: K
        int64_t nPhysB = mat2.size(1);  // physical N for B: N/2

        int64_t m = mat1.size(0);
        int64_t k = kPhysA * 2;  // logical K
        int64_t n = nPhysB * 2;  // logical N

        TORCH_CHECK(kPhysB == kPhysA * 2,
                    "w4a4_matmul: mat2.size(0) must equal mat1.size(1) * 2 (",
                    kPhysB, " vs ", kPhysA * 2, ")");
        TORCH_CHECK(scale.size(0) == n,
                    "w4a4_matmul: scale.size(0) must equal logical N (",
                    scale.size(0), " vs ", n, ")");
        TORCH_CHECK(perTokenScale.size(0) == m,
                    "w4a4_matmul: perTokenScale.size(0) must equal M (",
                    perTokenScale.size(0), " vs ", m, ")");

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(scale.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(perTokenScale.storage().data()));
    }

    static OutputType AllocOutput(
        const CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.m, params.n}, AclDtypeToTorchDtype(ACL_BF16));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& scale, const at::Tensor& perTokenScale,
        const c10::ScalarType& outDType, bool transA, bool transB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, scale, perTokenScale, outDType, transA, transB, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
