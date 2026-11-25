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

#ifndef OPTEST_SVD_QUANT_MATMUL_H
#define OPTEST_SVD_QUANT_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "mx_matmul.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using SvdQuantKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::SvdQuantMatmulParams&);

inline void CheckColumnMajor2D(const at::Tensor& tensor, const char* name, int64_t rows, int64_t cols)
{
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(
        tensor.size(0) == rows && tensor.size(1) == cols, name, " must have shape (", rows, ", ", cols, "), got (",
        tensor.size(0), ", ", tensor.size(1), ")");
    TORCH_CHECK(
        tensor.stride(0) == 1 && tensor.stride(1) == rows,
        name, " must use ColumnMajor storage layout with strides (1, ", rows, "), got (", tensor.stride(0), ", ",
        tensor.stride(1), ")");
}

template <SvdQuantKernelFn KernelFunc>
struct SvdQuantMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& x, const at::Tensor& svd1, const at::Tensor& svd2, const at::Tensor& w,
        const at::Tensor& w_scale, const at::Tensor& smooth_scale, double qmax,
        CatlassKernel::TParams& tParams, CatlassKernel::SvdQuantMatmulParams& params)
    {
        CheckNpuTensor(x, "x");
        CheckNpuTensor(svd1, "svd1");
        CheckNpuTensor(svd2, "svd2");
        CheckNpuTensor(w, "w");
        CheckNpuTensor(w_scale, "w_scale");
        CheckNpuTensor(smooth_scale, "smooth_scale");
        CheckSameDevice(x, "x", svd1, "svd1");
        CheckSameDevice(x, "x", svd2, "svd2");
        CheckSameDevice(x, "x", w, "w");
        CheckSameDevice(x, "x", w_scale, "w_scale");
        CheckSameDevice(x, "x", smooth_scale, "smooth_scale");
        CheckMxScaleDType(w_scale, "w_scale");

        TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must have dtype torch.float16, got ", x.scalar_type());
        TORCH_CHECK(svd1.scalar_type() == torch::kFloat16, "svd1 must have dtype torch.float16");
        TORCH_CHECK(svd2.scalar_type() == torch::kFloat16, "svd2 must have dtype torch.float16");
        TORCH_CHECK(smooth_scale.scalar_type() == torch::kFloat16, "smooth_scale must have dtype torch.float16");
        TORCH_CHECK(x.dim() == 2 && svd1.dim() == 2 && svd2.dim() == 2, "x, svd1, svd2 must be 2-D tensors");
        TORCH_CHECK(
            w.scalar_type() == torch::kInt8,
            "w must be int8 packed FP4 bytes (ColumnMajor KxN layout), got ", w.scalar_type());
        TORCH_CHECK(w.dim() == 1, "w must be a 1-D int8 tensor of packed FP4 bytes");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
        TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous");
        TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");

        const int64_t m = x.size(0);
        const int64_t k = x.size(1);
        TORCH_CHECK(svd1.size(0) == k, "svd1 first dimension must equal K");
        const int64_t r = svd1.size(1);
        TORCH_CHECK(svd2.size(0) == r, "svd2 first dimension must equal R");
        const int64_t n = svd2.size(1);
        CheckColumnMajor2D(svd1, "svd1", k, r);
        CheckColumnMajor2D(svd2, "svd2", r, n);
        const int64_t packedBytes = (k * n + 1) / 2;
        TORCH_CHECK(
            w.numel() == packedBytes,
            "w packed bytes must have ", packedBytes, " elements (k=", k, ", n=", n, "), got ", w.numel());
        TORCH_CHECK(qmax > 0.0, "qmax must be positive");

        tParams.element["X"] = ACL_FLOAT16;
        tParams.element["Y"] = ACL_FLOAT16;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);
        params.r = static_cast<uint32_t>(r);
        params.qmax = static_cast<float>(qmax);

        uint32_t mxScaleAlignedK = 0;
        int64_t scaleBNumel = 0;
        int64_t unused = 0;
        ComputeMxScaleShapes(params.m, params.n, params.k, mxScaleAlignedK, unused, scaleBNumel);
        (void)unused;
        TORCH_CHECK(
            w_scale.numel() == scaleBNumel,
            "w_scale must have ", scaleBNumel, " elements (n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", w_scale.numel());
        TORCH_CHECK(
            smooth_scale.numel() == k,
            "smooth_scale must have ", k, " elements, got ", smooth_scale.numel());

        params.inputAddr.resize(6);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(x.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(svd1.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(svd2.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(w.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(w_scale.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(smooth_scale.storage().data()));
    }

    static OutputType AllocOutput(CatlassKernel::SvdQuantMatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat16);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& x, const at::Tensor& svd1, const at::Tensor& svd2, const at::Tensor& w,
        const at::Tensor& w_scale, const at::Tensor& smooth_scale, double qmax)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::SvdQuantMatmulParams params;
        GetKernelInfo(x, svd1, svd2, w, w_scale, smooth_scale, qmax, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
