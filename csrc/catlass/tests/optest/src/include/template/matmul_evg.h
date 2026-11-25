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

#ifndef OPTEST_MATMUL_EVG_H
#define OPTEST_MATMUL_EVG_H

#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using EvgKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulEvgParams&);

namespace detail {

inline void FillAscend950EvgTParams(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, bool transA, bool transB,
    bool formatA, bool formatB, CatlassKernel::TParams& tParams)
{
    auto aclType = TorchDtypeToAclDtype(mat1.scalar_type());
    tParams.element["A"] = aclType;
    tParams.element["B"] = aclType;
    tParams.element["C"] = TorchDtypeToAclDtype(outDType);
    tParams.transpose["A"] = transA;
    tParams.transpose["B"] = transB;
    tParams.transpose["C"] = false;
    tParams.useNz["A"] = formatA;
    tParams.useNz["B"] = formatB;
    tParams.useNz["C"] = false;
}

inline void FillMatmulShapes(
    const at::Tensor& mat1, const at::Tensor& mat2, bool transA, bool transB, CatlassKernel::MatmulEvgParams& params)
{
    int64_t m, k1, k2, n;
    if (transA) {
        m = mat1.size(1);
        k1 = mat1.size(0);
    } else {
        m = mat1.size(0);
        k1 = mat1.size(1);
    }
    if (transB) {
        k2 = mat2.size(1);
        n = mat2.size(0);
    } else {
        k2 = mat2.size(0);
        n = mat2.size(1);
    }
    TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied (", m, "x", k1, " and ", k2, "x", n, ")");
    params.m = static_cast<uint32_t>(m);
    params.k = static_cast<uint32_t>(k1);
    params.n = static_cast<uint32_t>(n);
}

inline at::Tensor AllocEvgOutput(const CatlassKernel::TParams& tParams, CatlassKernel::MatmulEvgParams& params)
{
    at::Tensor output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
    params.outputAddr.resize(1);
    params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
    return output;
}

inline bool NeedsEvgExtra(const std::string& evgType)
{
    return evgType == "add" || evgType == "add_ub" || evgType == "bias";
}

inline void ValidateAddUbConstraints(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& aux, const c10::ScalarType& outDType,
    bool transA, bool transB, bool formatA, bool formatB)
{
    TORCH_CHECK(
        !transA && !transB,
        "matmul_evg evgType=add_ub (L0C->UB path) only supports RowMajor A/B; set transA=transB=false");
    TORCH_CHECK(!formatA && !formatB, "matmul_evg evgType=add_ub does not support NZ format tensors");
    TORCH_CHECK(
        mat1.scalar_type() == mat2.scalar_type(),
        "matmul_evg evgType=add_ub requires mat1 and mat2 to have the same dtype");
    TORCH_CHECK(
        aux.scalar_type() == outDType,
        "matmul_evg evgType=add_ub requires extra dtype to match outDType (e.g. fp16 inputs with fp32 output/extra)");
    TORCH_CHECK(
        outDType == torch::kFloat32,
        "matmul_evg evgType=add_ub (L0C->UB path) only supports fp32 output; fp16 output requires "
        "NO_SPLIT Fixpipe and is not supported in optest");
}

inline void BindEvgExtra(
    const std::string& evgType, const at::Tensor& extra, CatlassKernel::MatmulEvgParams& params)
{
    TORCH_CHECK(extra.defined() && extra.numel() > 0, "matmul_evg evgType=", evgType, " requires a non-empty extra tensor");
    TORCH_CHECK(extra.is_contiguous(), "extra must be contiguous");

    if (evgType == "add" || evgType == "add_ub") {
        TORCH_CHECK(extra.dim() == 2, "extra must be a 2-D tensor for evgType=", evgType);
        TORCH_CHECK(
            extra.size(0) == static_cast<int64_t>(params.m) && extra.size(1) == static_cast<int64_t>(params.n),
            "extra shape must match output shape (", params.m, ", ", params.n, "), got (", extra.size(0), ", ",
            extra.size(1), ")");
    } else if (evgType == "bias") {
        TORCH_CHECK(extra.dim() == 1, "extra must be a 1-D row bias vector for evgType=bias");
        TORCH_CHECK(
            extra.size(0) == static_cast<int64_t>(params.n),
            "bias length must equal N dimension (", params.n, "), got ", extra.size(0));
    }

    params.inputAddr.resize(3);
    params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(extra.storage().data()));
}

inline void LaunchEvgKernel(
    EvgKernelFn kernelFunc, CatlassKernel::TParams& tParams, CatlassKernel::MatmulEvgParams& params)
{
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    RUN_NPU_FUNC(kernelFunc, aicCoreNum, stream, tParams, params);
}

} // namespace detail

template <EvgKernelFn KernelFunc>
struct MatmulEvgLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& extra, const c10::ScalarType& outDType,
        const std::string& evgType, double negativeSlope, bool transA, bool transB, bool formatA, bool formatB)
    {
        TORCH_CHECK(!evgType.empty(), "evgType must not be empty");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");

        if (evgType == "add_ub") {
            detail::ValidateAddUbConstraints(mat1, mat2, extra, outDType, transA, transB, formatA, formatB);
        }

        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulEvgParams params;
        detail::FillAscend950EvgTParams(mat1, mat2, outDType, transA, transB, formatA, formatB, tParams);
        detail::FillMatmulShapes(mat1, mat2, transA, transB, params);
        params.evgType = evgType;
        params.negativeSlope = static_cast<float>(negativeSlope);

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        if (detail::NeedsEvgExtra(evgType)) {
            detail::BindEvgExtra(evgType, extra, params);
        }

        OutputType output = detail::AllocEvgOutput(tParams, params);
        detail::LaunchEvgKernel(KernelFunc, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
