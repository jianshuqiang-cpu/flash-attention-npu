#ifndef OPTEST_GROUPED_FIXPIPE_DEQUANT_MATMUL_H
#define OPTEST_GROUPED_FIXPIPE_DEQUANT_MATMUL_H

#include <cstring>
#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

#include "template/grouped_matmul.h"

namespace CatlassKernelWrapper {

template <GroupedKernelFn KernelFunc>
struct GroupedFixpipeDequantMatmulLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList, const at::Tensor& perChannelScale,
        double scalePerTensor, int64_t quantMode,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool useNzA, bool useNzB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GroupedMatmulParams params;

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.element["QUANT_MODE"] = static_cast<aclDataType>(quantMode);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = useNzA;
        tParams.useNz["B"] = useNzB;
        tParams.useNz["C"] = false;

        auto mat1Sizes = mat1.sizes();
        auto mat2Sizes = mat2.sizes();
        auto g = static_cast<uint32_t>(groupList.numel());

        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D (M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (G, K, N)");
        int64_t M = transA ? mat1Sizes[1] : mat1Sizes[0];
        int64_t k1 = transA ? mat1Sizes[0] : mat1Sizes[1];
        int64_t G = mat2Sizes[0];
        int64_t k2 = transB ? mat2Sizes[2] : mat2Sizes[1];
        int64_t N = transB ? mat2Sizes[1] : mat2Sizes[2];
        TORCH_CHECK(G == static_cast<int64_t>(g), "mat2[0] must equal groupList size");
        TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch");

        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(perChannelScale.storage().data()));

        float scalePT = static_cast<float>(scalePerTensor);
        uint8_t* scaleEncoded = nullptr;
        std::memcpy(&scaleEncoded, &scalePT, sizeof(float));
        params.inputAddr[4] = scaleEncoded;

        params.m = static_cast<uint32_t>(M);
        params.n = static_cast<uint32_t>(N);
        params.k = static_cast<uint32_t>(k1);
        params.batch = g;
        params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::M;

        OutputType output = GetOutputTensor(
            {params.m, params.n}, static_cast<torch::Dtype>(outDType));
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
