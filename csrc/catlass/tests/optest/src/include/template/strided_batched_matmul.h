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

#ifndef OPTEST_STRIDED_BATCHED_MATMUL_H
#define OPTEST_STRIDED_BATCHED_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using StridedBatchedKernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::StridedBatchedMatmulParams&);

template <StridedBatchedKernelFn KernelFunc>
struct StridedBatchedMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        bool batchTransA, bool batchTransB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::StridedBatchedMatmulParams& params)
    {
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = formatA;
        tParams.useNz["B"] = formatB;
        tParams.useNz["C"] = false;

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        int64_t btransA = batchTransA ? 1 : 0;
        int64_t btransB = batchTransB ? 1 : 0;

        // Shape extraction matches example: logical shape is always (batch, M, K) for A,
        // (batch, K, N) for B, regardless of RowMajor/ColumnMajor layout.
        // When ColumnMajor, the physical storage has M/K (or K/N) swapped, so we swap
        // the dimension extraction accordingly.
        int64_t batch = mat1.dim() >= 3 ? mat1.size(btransA) : 1;
        int64_t m, k;
        if (!transA) {
            m = mat1.size(mat1.dim() - 2 - btransA);
            k = mat1.size(mat1.dim() - 1);
        } else {
            k = mat1.size(mat1.dim() - 2 - btransA);
            m = mat1.size(mat1.dim() - 1);
        }
        int64_t k2, n;
        if (!transB) {
            k2 = mat2.size(mat2.dim() - 2 - btransB);
            n = mat2.size(mat2.dim() - 1);
        } else {
            n = mat2.size(mat2.dim() - 2 - btransB);
            k2 = mat2.size(mat2.dim() - 1);
        }
        TORCH_CHECK(k == k2, "mat1 and mat2 shapes cannot be multiplied (",
                    m, "x", k, " and ", k2, "x", n, ")");
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);
        params.batch = static_cast<uint32_t>(batch);

        // leading dimension: stride for the non-batch dimension
        // strideA: stride for the batch dimension
        if (!transA) {  // RowMajor
            params.lda = batchTransA ? static_cast<int64_t>(batch) * k : static_cast<int64_t>(k);
            params.strideA = batchTransA ? static_cast<int64_t>(k) : static_cast<int64_t>(m) * params.lda;
        } else {  // ColumnMajor
            params.lda = batchTransA ? static_cast<int64_t>(batch) * m : static_cast<int64_t>(m);
            params.strideA = batchTransA ? static_cast<int64_t>(m) : static_cast<int64_t>(k) * params.lda;
        }

        if (!transB) {  // RowMajor
            params.ldb = batchTransB ? static_cast<int64_t>(batch) * n : static_cast<int64_t>(n);
            params.strideB = batchTransB ? static_cast<int64_t>(n) : static_cast<int64_t>(k) * params.ldb;
        } else {  // ColumnMajor
            params.ldb = batchTransB ? static_cast<int64_t>(batch) * k : static_cast<int64_t>(k);
            params.strideB = batchTransB ? static_cast<int64_t>(k) : static_cast<int64_t>(n) * params.ldb;
        }

        params.ldc = static_cast<int64_t>(n);
        params.strideC = static_cast<int64_t>(m) * params.ldc;
    }

    static OutputType AllocOutput(
        const CatlassKernel::TParams& tParams, CatlassKernel::StridedBatchedMatmulParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        bool batchTransA, bool batchTransB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::StridedBatchedMatmulParams params;
        GetKernelInfo(mat1, mat2, outDType, transA, transB, formatA, formatB,
                      batchTransA, batchTransB, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
