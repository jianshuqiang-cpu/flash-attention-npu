#ifndef OPTEST_GROUPED_MATMUL_H
#define OPTEST_GROUPED_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using GroupedKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::GroupedMatmulParams&);

enum class GmmSliceDir { M, K };

template <GroupedKernelFn KernelFunc, GmmSliceDir SliceDir>
struct GroupedMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& groupOnDevice,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool useNzA, bool useNzB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::GroupedMatmulParams& params)
    {
        auto aclType = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["A"] = aclType;
        tParams.element["B"] = aclType;
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = useNzA;
        tParams.useNz["B"] = useNzB;
        tParams.useNz["C"] = false;

        auto g = static_cast<uint32_t>(groupOnDevice.numel());
        auto mat1Sizes = mat1.sizes();
        auto mat2Sizes = mat2.sizes();

        if constexpr (SliceDir == GmmSliceDir::M) {
            TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D (M, K) for slice-M");
            TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (G, K, N) for slice-M");
            int64_t M, k1;
            if (transA) { k1 = mat1Sizes[0]; M = mat1Sizes[1]; }
            else        { M = mat1Sizes[0]; k1 = mat1Sizes[1]; }
            int64_t G = mat2Sizes[0];
            int64_t k2 = transB ? mat2Sizes[2] : mat2Sizes[1];
            int64_t N = transB ? mat2Sizes[1] : mat2Sizes[2];
            TORCH_CHECK(G == static_cast<int64_t>(g), "mat2[0] must equal groupList size");
            TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch");

            params.inputAddr.resize(3);
            params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
            params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
            params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupOnDevice.storage().data()));
            params.m = static_cast<uint32_t>(M);
            params.n = static_cast<uint32_t>(N);
            params.k = static_cast<uint32_t>(k1);
            params.batch = g;
            params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::M;
        } else {
            TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D for slice-K");
            TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2-D (K, N) for slice-K");
            int64_t k1, M, K2, N;
            if (transA) {
                k1 = mat1Sizes[0]; M = mat1Sizes[1];
            } else {
                M = mat1Sizes[0]; k1 = mat1Sizes[1];
            }
            K2 = mat2Sizes[0];
            N = mat2Sizes[1];
            TORCH_CHECK(k1 == K2, "mat1 and mat2 k dim mismatch");

            params.inputAddr.resize(3);
            params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
            params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
            params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupOnDevice.storage().data()));
            params.m = static_cast<uint32_t>(M);
            params.n = static_cast<uint32_t>(N);
            params.k = static_cast<uint32_t>(k1);
            params.batch = g;
            params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::K;
        }
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& groupList,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool useNzA, bool useNzB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GroupedMatmulParams params;
        TORCH_CHECK(groupList.dtype() == torch::kInt64, "groupList must be int64");
        GetKernelInfo(mat1, mat2, groupList, outDType, transA, transB, useNzA, useNzB, tParams, params);

        OutputType output;
        if constexpr (SliceDir == GmmSliceDir::M) {
            output = GetOutputTensor(
                {params.m, params.n}, static_cast<torch::Dtype>(outDType));
        } else {
            output = GetOutputTensor(
                {static_cast<int64_t>(params.batch), params.m, params.n},
                static_cast<torch::Dtype>(outDType));
        }
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
