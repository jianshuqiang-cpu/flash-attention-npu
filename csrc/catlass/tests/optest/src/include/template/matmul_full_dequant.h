/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_MATMUL_FULL_DEQUANT_H
#define OPTEST_MATMUL_FULL_DEQUANT_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

enum class X1QuantMode { kDefault = 0, kPerTensor = 1, kPerToken = 4 };
enum class X2QuantMode { kDefault = 0, kPerTensor = 1, kPerChannel = 2 };

template <KernelFn KernelFunc>
struct MatmulFullDequantLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::optional<at::Tensor>& x1Scale, const c10::optional<at::Tensor>& x2Scale,
        const c10::optional<at::Tensor>& bias,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        X1QuantMode x1QuantMode, X2QuantMode x2QuantMode,
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
        tParams.element["D"] = TorchDtypeToAclDtype(outDType);
        params.x1QuantMode = static_cast<uint32_t>(x1QuantMode);
        params.x2QuantMode = static_cast<uint32_t>(x2QuantMode);
        params.hasQuantBias = bias.has_value();

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

        int inputIdx = 2;
        if (x1Scale.has_value()) {
            params.inputAddr.resize(inputIdx + 1);
            params.inputAddr[inputIdx] = static_cast<uint8_t*>(const_cast<void*>(x1Scale->storage().data()));
            inputIdx++;
        }
        if (x2Scale.has_value()) {
            params.inputAddr.resize(inputIdx + 1);
            params.inputAddr[inputIdx] = static_cast<uint8_t*>(const_cast<void*>(x2Scale->storage().data()));
            inputIdx++;
        }
        if (bias.has_value()) {
            params.inputAddr.resize(inputIdx + 1);
            params.inputAddr[inputIdx] = static_cast<uint8_t*>(const_cast<void*>(bias->storage().data()));
            inputIdx++;
        }

        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
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
        const c10::optional<at::Tensor>& x1Scale, const c10::optional<at::Tensor>& x2Scale,
        const c10::optional<at::Tensor>& bias,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        const std::string& x1QuantModeStr, const std::string& x2QuantModeStr)
    {
        X1QuantMode x1qm = X1QuantMode::kDefault;
        if (x1QuantModeStr == "per_tensor") x1qm = X1QuantMode::kPerTensor;
        else if (x1QuantModeStr == "per_token") x1qm = X1QuantMode::kPerToken;

        X2QuantMode x2qm = X2QuantMode::kDefault;
        if (x2QuantModeStr == "per_tensor") x2qm = X2QuantMode::kPerTensor;
        else if (x2QuantModeStr == "per_channel") x2qm = X2QuantMode::kPerChannel;

        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        params.inputAddr.resize(2);
        GetKernelInfo(mat1, mat2, x1Scale, x2Scale, bias, outDType, transA, transB, formatA, formatB, x1qm, x2qm, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
