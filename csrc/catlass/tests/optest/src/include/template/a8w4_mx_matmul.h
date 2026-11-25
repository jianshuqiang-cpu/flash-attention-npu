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

#ifndef OPTEST_A8W4_MX_MATMUL_H
#define OPTEST_A8W4_MX_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "mx_matmul.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

/**
 * @brief Adapter for example 59_ascend950_a8w4_mx_matmul (A8 FP8 x W4 FP4 MX matmul).
 *
 * Follows the example layout: transA=false, prologue B ColumnMajor (K, N).
 */
template <KernelFn KernelFunc>
struct A8W4MxMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(mx_scale_a, "mx_scale_a");
        CheckNpuTensor(mx_scale_b, "mx_scale_b");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        CheckSameDevice(mat1, "mat1", mx_scale_a, "mx_scale_a");
        CheckSameDevice(mat1, "mat1", mx_scale_b, "mx_scale_b");
        CheckMxScaleDType(mx_scale_a, "mx_scale_a");
        CheckMxScaleDType(mx_scale_b, "mx_scale_b");

        TORCH_CHECK(
            mat1.scalar_type() == torch::kFloat8_e4m3fn, "mat1 must have dtype torch.float8_e4m3fn, got ",
            mat1.scalar_type());
        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D with shape (M, K)");
        TORCH_CHECK(
            mat2.scalar_type() == torch::kInt8,
            "mat2 must be int8 packed FP4 prologue bytes (ColumnMajor KxN layout), got ", mat2.scalar_type());
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");

        const int64_t m = mat1.size(0);
        const int64_t k = mat1.size(1);
        int64_t n = 0;
        if (mx_scale_b.dim() == 3) {
            n = mx_scale_b.size(0);
        } else {
            TORCH_CHECK(false, "mx_scale_b must have shape (N, mxScaleAlignedK/2, 2) for A8W4 MX matmul");
        }
        const int64_t packedBytes = (k * n + 1) / 2;
        TORCH_CHECK(
            mat2.numel() == packedBytes,
            "mat2 packed bytes must have ", packedBytes, " elements (k=", k, ", n=", n, "), got ", mat2.numel());

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = test_utils::TypeCast<std::string, aclDataType>("float4_e2m1fn_x2");
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;
        tParams.l1TileShape = {128, 128, 128};
        tParams.l0TileShape = {128, 128, 128};
        tParams.swizzle = {3, 0, 0};

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);
        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));

        uint32_t mxScaleAlignedK = 0;
        int64_t scaleANumel = 0;
        int64_t scaleBNumel = 0;
        ComputeMxScaleShapes(params.m, params.n, params.k, mxScaleAlignedK, scaleANumel, scaleBNumel);
        TORCH_CHECK(
            mx_scale_a.numel() == scaleANumel,
            "mx_scale_a must have ", scaleANumel, " elements (m=", m, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_a.numel());
        TORCH_CHECK(
            mx_scale_b.numel() == scaleBNumel,
            "mx_scale_b must have ", scaleBNumel, " elements (n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_b.numel());
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat32);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, mx_scale_a, mx_scale_b, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
