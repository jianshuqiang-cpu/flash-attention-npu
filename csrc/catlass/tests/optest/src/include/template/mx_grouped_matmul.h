#ifndef OPTEST_MX_GROUPED_MATMUL_H
#define OPTEST_MX_GROUPED_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"
#include "template/mx_matmul.h"
#include "template/grouped_matmul.h"

namespace CatlassKernelWrapper {

using MxGroupedKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::GroupedMatmulParams&);

template <MxGroupedKernelFn KernelFunc>
struct MxGroupedMatmulLike {
    using OutputType = at::Tensor;

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList,
        const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
        bool transA, bool transB,
        bool enableAswt, bool enablePreload)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GroupedMatmulParams params;

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
        tParams.flag["ASWT"] = enableAswt;
        tParams.flag["PRELOAD"] = enablePreload;

        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D (M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (G, K, N)");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");

        auto g = static_cast<uint32_t>(groupList.numel());
        int64_t M = transA ? mat1.size(1) : mat1.size(0);
        int64_t k1 = transA ? mat1.size(0) : mat1.size(1);
        int64_t G = mat2.size(0);
        int64_t k2 = transB ? mat2.size(2) : mat2.size(1);
        int64_t N = transB ? mat2.size(1) : mat2.size(2);
        TORCH_CHECK(G == static_cast<int64_t>(g), "mat2[0] must equal groupList size");
        TORCH_CHECK(k1 == k2, "mat1 and mat2 k dim mismatch");

        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));
        params.m = static_cast<uint32_t>(M);
        params.n = static_cast<uint32_t>(N);
        params.k = static_cast<uint32_t>(k1);
        params.batch = g;
        params.sliceMode = CatlassKernel::GroupedMatmulParams::SliceMode::M;

        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat32);
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
