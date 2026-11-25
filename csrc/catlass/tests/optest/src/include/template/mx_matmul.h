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

#ifndef OPTEST_MX_MATMUL_H
#define OPTEST_MX_MATMUL_H

#include <tuple>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

constexpr uint32_t kMxScaleGroupNum = 32;

inline uint32_t CeilDivUint32(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

inline uint32_t RoundUp2Uint32(uint32_t v)
{
    return (v + 1U) / 2U * 2U;
}

/**
 * @brief Compute MX scale tensor shapes for the default layout mapping.
 *
 * When A is RowMajor and B is ColumnMajor (trans_a=0, trans_b=1):
 *   mx_scale_a: (m, mxScaleAlignedK / 2, 2)  → numel = m * mxScaleAlignedK
 *   mx_scale_b: (n, mxScaleAlignedK / 2, 2)  → numel = n * mxScaleAlignedK
 */
inline void ComputeMxScaleShapes(
    uint32_t m, uint32_t n, uint32_t k, uint32_t& mxScaleAlignedK, int64_t& scaleANumel, int64_t& scaleBNumel)
{
    const uint32_t mxScaleK = CeilDivUint32(k, kMxScaleGroupNum);
    mxScaleAlignedK = RoundUp2Uint32(mxScaleK);
    scaleANumel = static_cast<int64_t>(m) * static_cast<int64_t>(mxScaleAlignedK);
    scaleBNumel = static_cast<int64_t>(n) * static_cast<int64_t>(mxScaleAlignedK);
}

inline int64_t ProductInt64(uint32_t a, uint32_t b, uint32_t c)
{
    return static_cast<int64_t>(a) * static_cast<int64_t>(b) * static_cast<int64_t>(c);
}

inline void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on NPU");
}

inline void CheckSameDevice(const at::Tensor& reference, const char* referenceName, const at::Tensor& tensor,
    const char* tensorName)
{
    TORCH_CHECK(
        tensor.device() == reference.device(), tensorName, " must be on the same device as ", referenceName,
        " (got ", tensor.device(), " and ", reference.device(), ")");
}

inline void CheckMxScaleDType(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat8_e8m0fnu, name, " must have dtype torch.float8_e8m0fnu, got ",
        tensor.scalar_type());
}

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <KernelFn KernelFunc>
struct MxMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, bool transA, bool transB, CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
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

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));

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
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);

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
        const at::Tensor& mx_scale_b, bool transA, bool transB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, mx_scale_a, mx_scale_b, transA, transB, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

template <KernelFn KernelFunc>
struct MxBatchedMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, bool transA, bool transB, CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        TORCH_CHECK(mat1.dim() == 3, "mat1 must be a 3-D tensor with shape (batch, M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3-D tensor with shape (batch, N, K) when transB=True");
        TORCH_CHECK(
            !transA && transB,
            "ascend950_fp8_mx_batch_matmul currently supports only transA=false and transB=true");
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(mx_scale_a, "mx_scale_a");
        CheckNpuTensor(mx_scale_b, "mx_scale_b");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        CheckSameDevice(mat1, "mat1", mx_scale_a, "mx_scale_a");
        CheckSameDevice(mat1, "mat1", mx_scale_b, "mx_scale_b");
        TORCH_CHECK(
            mat1.scalar_type() == torch::kFloat8_e4m3fn, "mat1 must have dtype torch.float8_e4m3fn, got ",
            mat1.scalar_type());
        TORCH_CHECK(
            mat2.scalar_type() == torch::kFloat8_e4m3fn, "mat2 must have dtype torch.float8_e4m3fn, got ",
            mat2.scalar_type());
        CheckMxScaleDType(mx_scale_a, "mx_scale_a");
        CheckMxScaleDType(mx_scale_b, "mx_scale_b");
        TORCH_CHECK(mat1.size(0) == mat2.size(0), "mat1 and mat2 batch dimensions must match");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_BF16;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(2);
            k1 = mat1.size(1);
        } else {
            m = mat1.size(1);
            k1 = mat1.size(2);
        }
        if (transB) {
            k2 = mat2.size(2);
            n = mat2.size(1);
        } else {
            k2 = mat2.size(1);
            n = mat2.size(2);
        }
        TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied (", m, "x", k1, " and ", k2, "x", n, ")");

        params.batch = static_cast<uint32_t>(mat1.size(0));
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);

        uint32_t mxScaleAlignedK = 0;
        int64_t scaleAPerBatch = 0;
        int64_t scaleBPerBatch = 0;
        ComputeMxScaleShapes(params.m, params.n, params.k, mxScaleAlignedK, scaleAPerBatch, scaleBPerBatch);
        TORCH_CHECK(
            mx_scale_a.numel() == static_cast<int64_t>(params.batch) * scaleAPerBatch,
            "mx_scale_a must have ", static_cast<int64_t>(params.batch) * scaleAPerBatch,
            " elements (batch=", params.batch, ", m=", m, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_a.numel());
        TORCH_CHECK(
            mx_scale_b.numel() == static_cast<int64_t>(params.batch) * scaleBPerBatch,
            "mx_scale_b must have ", static_cast<int64_t>(params.batch) * scaleBPerBatch,
            " elements (batch=", params.batch, ", n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_b.numel());

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.batch, params.m, params.n}, torch::kBFloat16);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, bool transA, bool transB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, mx_scale_a, mx_scale_b, transA, transB, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

template <KernelFn KernelFunc>
struct DualLevelQuantMxBatchedMatmulLike {
    using OutputType =
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

    struct ScratchBundle {
        at::Tensor output;
        at::Tensor scaleA1;
        at::Tensor scaleA2;
        at::Tensor scaleB1;
        at::Tensor scaleB2;
        at::Tensor workspace;
    };

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        TORCH_CHECK(mat1.dim() == 3, "mat1 must be a 3-D tensor with shape (batch, M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be a 3-D tensor with shape (batch, N, K)");
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        TORCH_CHECK(
            mat1.scalar_type() == torch::kFloat16 || mat1.scalar_type() == torch::kBFloat16,
            "mat1 must have dtype torch.float16 or torch.bfloat16, got ", mat1.scalar_type());
        TORCH_CHECK(
            mat2.scalar_type() == mat1.scalar_type(), "mat2 dtype must match mat1 dtype (got ", mat2.scalar_type(),
            " and ", mat1.scalar_type(), ")");
        TORCH_CHECK(mat1.size(0) == mat2.size(0), "mat1 and mat2 batch dimensions must match");
        TORCH_CHECK(mat1.size(2) == mat2.size(2), "mat1 K and mat2 K dimensions must match");
        TORCH_CHECK(mat1.size(2) % 2 == 0, "K must be even for FP4 packing, got ", mat1.size(2));
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");

        tParams.element["INPUT"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["C"] = ACL_BF16;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.batch = static_cast<uint32_t>(mat1.size(0));
        params.m = static_cast<uint32_t>(mat1.size(1));
        params.k = static_cast<uint32_t>(mat1.size(2));
        params.n = static_cast<uint32_t>(mat2.size(1));
        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
    }

    static ScratchBundle AllocOutputAndScratch(CatlassKernel::MatmulParams& params)
    {
        ScratchBundle scratch;
        scratch.output = GetOutputTensor({params.batch, params.m, params.n}, torch::kBFloat16);
        const uint32_t scaleA1K = CeilDivUint32(params.k, 512);
        const uint32_t mxScaleAlignedK = RoundUp2Uint32(CeilDivUint32(params.k, kMxScaleGroupNum));

        scratch.scaleA1 = GetOutputTensor({params.batch, params.m, scaleA1K}, torch::kFloat32);
        scratch.scaleA2 = GetOutputTensor({params.batch, params.m, mxScaleAlignedK}, torch::kFloat8_e8m0fnu);
        scratch.scaleB1 = GetOutputTensor({params.batch, params.n, scaleA1K}, torch::kFloat32);
        scratch.scaleB2 = GetOutputTensor({params.batch, params.n, mxScaleAlignedK}, torch::kFloat8_e8m0fnu);
        scratch.workspace = GetOutputTensor({ProductInt64(params.batch, params.m, params.k / 2) +
                                             ProductInt64(params.batch, params.n, params.k / 2)}, torch::kUInt8);

        params.outputAddr.resize(6);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(scratch.output.storage().data()));
        params.outputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(scratch.scaleA1.storage().data()));
        params.outputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(scratch.scaleA2.storage().data()));
        params.outputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(scratch.scaleB1.storage().data()));
        params.outputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(scratch.scaleB2.storage().data()));
        params.outputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(scratch.workspace.storage().data()));
        return scratch;
    }

    static OutputType Run(const at::Tensor& mat1, const at::Tensor& mat2)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, tParams, params);
        ScratchBundle scratch = AllocOutputAndScratch(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return {
            scratch.output,
            scratch.scaleA1,
            scratch.scaleA2,
            scratch.scaleB1,
            scratch.scaleB2};
    }
};

using GroupedMxFinalizeRoutingKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&,
    const CatlassKernel::GroupedMxFinalizeRoutingParams&);

template <GroupedMxFinalizeRoutingKernelFn KernelFunc>
struct MxGroupedMatmulFinalizeRoutingLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1,
        const at::Tensor& mat2,
        const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b,
        const at::Tensor& group_list,
        const at::Tensor& logit,
        const at::Tensor& row_index,
        const at::Tensor& bias,
        const at::Tensor& shared_input,
        bool transA,
        bool transB,
        int64_t batch,
        int64_t data_parallel_size,
        double shared_input_weight,
        int64_t shared_input_offset,
        int64_t group_list_type)
    {
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(mx_scale_a, "mx_scale_a");
        CheckNpuTensor(mx_scale_b, "mx_scale_b");
        CheckNpuTensor(group_list, "group_list");
        CheckNpuTensor(logit, "logit");
        CheckNpuTensor(row_index, "row_index");
        TORCH_CHECK(mat1.scalar_type() == torch::kFloat8_e4m3fn || mat1.scalar_type() == torch::kFloat8_e5m2,
                    "mat1 must be float8_e4m3fn or float8_e5m2");
        TORCH_CHECK(mat2.scalar_type() == torch::kFloat8_e4m3fn || mat2.scalar_type() == torch::kFloat8_e5m2,
                    "mat2 must be float8_e4m3fn or float8_e5m2");
        CheckMxScaleDType(mx_scale_a, "mx_scale_a");
        CheckMxScaleDType(mx_scale_b, "mx_scale_b");
        TORCH_CHECK(group_list.dtype() == torch::kInt64, "group_list must be int64");
        TORCH_CHECK(logit.dtype() == torch::kFloat32, "logit must be float32");
        TORCH_CHECK(row_index.dtype() == torch::kInt64, "row_index must be int64");

        bool enableBias = bias.defined() && bias.numel() > 0;
        bool enableSharedInput = shared_input.defined() && shared_input.numel() > 0;
        if (enableBias) {
            CheckNpuTensor(bias, "bias");
            TORCH_CHECK(bias.dtype() == torch::kBFloat16, "bias must be bfloat16");
        }
        if (enableSharedInput) {
            CheckNpuTensor(shared_input, "shared_input");
            TORCH_CHECK(shared_input.dtype() == torch::kBFloat16, "shared_input must be bfloat16");
        }

        CatlassKernel::TParams tParams;
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(1);
            k1 = mat1.size(0);
        } else {
            m = mat1.size(0);
            k1 = mat1.size(1);
        }
        if (transB) {
            k2 = mat2.size(mat2.dim() - 1);
            n = mat2.size(mat2.dim() - 2);
        } else {
            k2 = mat2.size(mat2.dim() - 2);
            n = mat2.size(mat2.dim() - 1);
        }
        TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch (", k1, " vs ", k2, ")");

        uint32_t problemCount = static_cast<uint32_t>(group_list.numel());
        int64_t bsdp = batch / data_parallel_size;

        CatlassKernel::GroupedMxFinalizeRoutingParams params;
        params.m = static_cast<uint32_t>(m);
        params.n = static_cast<uint32_t>(n);
        params.k = static_cast<uint32_t>(k1);
        params.batch = static_cast<uint32_t>(batch);
        params.problemCount = problemCount;
        params.groupListType = static_cast<uint32_t>(group_list_type);
        params.sharedInputWeight = static_cast<float>(shared_input_weight);
        params.sharedInputOffset = static_cast<uint32_t>(shared_input_offset);
        params.bsdp = static_cast<uint32_t>(bsdp);

        params.inputAddr.resize(9);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(group_list.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(logit.storage().data()));
        params.inputAddr[6] = static_cast<uint8_t*>(const_cast<void*>(row_index.storage().data()));
        params.inputAddr[7] = enableBias
            ? static_cast<uint8_t*>(const_cast<void*>(bias.storage().data()))
            : nullptr;
        params.inputAddr[8] = enableSharedInput
            ? static_cast<uint8_t*>(const_cast<void*>(shared_input.storage().data()))
            : nullptr;

        OutputType output = GetOutputTensor({static_cast<int64_t>(batch), static_cast<int64_t>(n)}, torch::kFloat32);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
