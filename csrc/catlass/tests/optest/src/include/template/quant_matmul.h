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

#ifndef OPTEST_QUANT_MATMUL_H
#define OPTEST_QUANT_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <KernelFn KernelFunc>
struct QuantMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& scale, const at::Tensor& perTokenScale,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_INT32;
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = formatA;
        tParams.useNz["B"] = formatB;
        tParams.useNz["C"] = false;
        tParams.element["SCALE"] = TorchDtypeToAclDtype(scale.scalar_type());
        tParams.element["PER_TOKEN_SCALE"] = TorchDtypeToAclDtype(perTokenScale.scalar_type());
        tParams.element["D"] = TorchDtypeToAclDtype(outDType);

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(scale.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(perTokenScale.storage().data()));

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(1);  k1 = mat1.size(0);
        } else {
            m = mat1.size(0);  k1 = mat1.size(1);
        }
        if (transB) {
            k2 = mat2.size(1); n = mat2.size(0);
        } else {
            k2 = mat2.size(0); n = mat2.size(1);
        }
        TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied (",
                    m, "x", k1, " and ", k2, "x", n, ")");
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);
    }

    static OutputType AllocOutput(
        const CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("D")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& scale, const at::Tensor& perTokenScale,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, scale, perTokenScale, outDType, transA, transB, formatA, formatB, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
