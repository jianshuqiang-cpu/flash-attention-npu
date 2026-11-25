#ifndef OPTEST_GROUP_GEMM_H
#define OPTEST_GROUP_GEMM_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using GemmKernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::GemmParams&);

template <GemmKernelFn KernelFunc>
struct GroupGemmLike {
    using OutputType = at::Tensor;
    static OutputType Run(const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList, const c10::ScalarType& outDType,
        double alpha, double beta, bool transA, bool transB, bool useNzA, bool useNzB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GemmParams params;
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = transA; tParams.transpose["B"] = transB;
        tParams.useNz["A"] = useNzA; tParams.useNz["B"] = useNzB;

        TORCH_CHECK(groupList.dtype() == torch::kInt64, "groupList must be int64");
        int64_t M, k1, K2, N;
        if (transA) { k1 = mat1.size(0); M = mat1.size(1); }
        else        { M = mat1.size(0); k1 = mat1.size(1); }
        K2 = mat2.size(0); N = mat2.size(1);
        TORCH_CHECK(k1 == K2, "K dim mismatch");
        params.batch = static_cast<uint32_t>(groupList.numel());
        params.m = static_cast<uint32_t>(M);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(N);
        params.alpha = static_cast<float>(alpha);
        params.beta  = static_cast<float>(beta);
        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));

        OutputType output = GetOutputTensor({static_cast<int64_t>(params.batch), params.m, params.n},
            AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, coreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper
#endif
