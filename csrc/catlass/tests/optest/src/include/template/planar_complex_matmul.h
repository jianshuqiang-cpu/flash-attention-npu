/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_PLANAR_COMPLEX_MATMUL_H
#define OPTEST_PLANAR_COMPLEX_MATMUL_H

#include <tuple>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <KernelFn KernelFunc>
struct PlanarComplexMatmulLike {
    using OutputType = std::tuple<at::Tensor, at::Tensor>;

    static void GetKernelInfo(
        const at::Tensor& aReal, const at::Tensor& aImag, const at::Tensor& bReal, const at::Tensor& bImag,
        CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        TORCH_CHECK(
            aReal.dim() == 2 && aImag.dim() == 2 && bReal.dim() == 2 && bImag.dim() == 2,
            "planar_complex_matmul expects four 2D tensors");
        TORCH_CHECK(aReal.sizes() == aImag.sizes(), "A real and imag tensors must have the same shape");
        TORCH_CHECK(bReal.sizes() == bImag.sizes(), "B real and imag tensors must have the same shape");
        TORCH_CHECK(aReal.size(1) == bReal.size(1), "A and B shapes cannot be multiplied");
        TORCH_CHECK(
            aReal.scalar_type() == torch::kFloat16 && aImag.scalar_type() == torch::kFloat16 &&
                bReal.scalar_type() == torch::kFloat16 && bImag.scalar_type() == torch::kFloat16,
            "planar_complex_matmul expects float16 inputs");

        uint32_t m = static_cast<uint32_t>(aReal.size(0));
        uint32_t k = static_cast<uint32_t>(aReal.size(1));
        uint32_t n = static_cast<uint32_t>(bReal.size(0));

        tParams.element["A"] = TorchDtypeToAclDtype(aReal.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(bReal.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        uint32_t coreLoops = (m + 127) / 128 * ((n + 255) / 256);
        double perCore = static_cast<double>(coreLoops) / aicCoreNum;
        tParams.flag["USE_FOUR_PASS"] = k >= 6000 && perCore >= 3.0;
        tParams.flag["NEGATE_A"] = m < n;
        tParams.flag["SWIZZLE_DIR_0"] = m >= n;

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(aReal.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(aImag.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(bReal.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(bImag.storage().data()));
        params.m = m;
        params.n = n;
        params.k = k;
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params, const at::Tensor& refTensor)
    {
        auto real = at::empty({params.m, params.n}, refTensor.options().dtype(torch::kFloat32));
        auto imag = at::empty({params.m, params.n}, refTensor.options().dtype(torch::kFloat32));
        params.outputAddr.resize(2);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(real.storage().data()));
        params.outputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(imag.storage().data()));
        return {real, imag};
    }

    static OutputType Run(
        const at::Tensor& aReal, const at::Tensor& aImag, const at::Tensor& bReal, const at::Tensor& bImag)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(aReal, aImag, bReal, bImag, tParams, params);
        auto output = AllocOutput(params, aReal);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
