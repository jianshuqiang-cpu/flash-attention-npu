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
 
#ifndef OPTEST_GROUPED_MX_SWIGLU_QUANT_MATMUL_H
#define OPTEST_GROUPED_MX_SWIGLU_QUANT_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

#include "template/grouped_matmul.h"
#include "template/mx_matmul.h"

namespace CatlassKernelWrapper {

template <GroupedKernelFn KernelFunc>
struct GroupedMxSwigluMxQuantMatmulLike {
    using OutputType = std::vector<at::Tensor>;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
        const at::Tensor& groupList, bool transB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::GroupedMatmulParams& params)
    {
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.element["Q"] = tParams.element["A"];
        tParams.element["Q_SCALE"] = tParams.element["MX_SCALE"];
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D (M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (G, N, K) for transB or (G, K, N)");
        TORCH_CHECK(groupList.dtype() == torch::kInt64, "groupList must be int64");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");

        auto g = static_cast<uint32_t>(groupList.numel());
        int64_t M, k1, k2, N;
        M = mat1.size(0);
        k1 = mat1.size(1);
        if (transB) {
            k2 = mat2.size(2);
            N = mat2.size(1);
        } else {
            k2 = mat2.size(1);
            N = mat2.size(2);
        }
        TORCH_CHECK(static_cast<int64_t>(g) == mat2.size(0), "groupList size must match mat2 batch dim");
        TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch");
        TORCH_CHECK(N % 128 == 0, "N must be 128-aligned");
        TORCH_CHECK(g <= 1024, "g must not exceed 1024");

        params.m = static_cast<uint32_t>(M);
        params.n = static_cast<uint32_t>(N);
        params.k = static_cast<uint32_t>(k1);
        params.batch = g;
        params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::M;

        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));
    }

    static OutputType AllocOutputs(
        const CatlassKernel::TParams& tParams, CatlassKernel::GroupedMatmulParams& params)
    {
        uint32_t M = params.m;
        uint32_t N = params.n;
        uint32_t N_half = N / 2;

        uint32_t mxScaleK_q = CeilDivUint32(N_half, kMxScaleGroupNum);
        int64_t qScaleRowStride = static_cast<int64_t>(mxScaleK_q);

        auto qTensor = GetOutputTensor(
            {static_cast<int64_t>(M), static_cast<int64_t>(N_half)},
            AclDtypeToTorchDtype(tParams.elem("Q")));
        auto qScaleTensor = GetOutputTensor(
            {static_cast<int64_t>(M), qScaleRowStride},
            AclDtypeToTorchDtype(tParams.elem("Q_SCALE")));

        params.outputAddr.resize(2);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(qTensor.storage().data()));
        params.outputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(qScaleTensor.storage().data()));

        return {qTensor, qScaleTensor};
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
        const at::Tensor& groupList, bool transB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GroupedMatmulParams params;
        GetKernelInfo(mat1, mat2, mx_scale_a, mx_scale_b, groupList, transB, tParams, params);
        OutputType outputs = AllocOutputs(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return outputs;
    }
};

} // namespace CatlassKernelWrapper

#endif
