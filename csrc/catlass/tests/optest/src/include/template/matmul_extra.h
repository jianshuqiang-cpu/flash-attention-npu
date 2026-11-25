#ifndef OPTEST_MATMUL_EXTRA_H
#define OPTEST_MATMUL_EXTRA_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

/**
 * @brief Unified adapter for matmul ops with an extra input tensor.
 *
 * Two modes selected by the compile-time ``IsBias`` flag:
 *
 * - ``IsBias == false`` (MatmulAdd, in-place):
 *   ``extra`` is the addend C, shape ``(M,N)``, dtype = outDType.
 *   C and output D share the same buffer — C is used directly as the output,
 *   no separate allocation or copy. The kernel reads C as addend and
 *   accumulates ``A @ B`` into C. Returns C (modified in-place).
 *   ``inputAddr = [A, B]``, ``outputAddr = [C]``.
 *
 * - ``IsBias == true``  (MatmulBias, out-of-place):
 *   ``extra`` is a 1-D bias vector, shape ``(N,)``, separate from output.
 *   Allocates a fresh output tensor.
 *   ``inputAddr = [A, B, bias]``, ``outputAddr = [output]``.
 */
template <KernelFn KernelFunc, bool IsBias>
struct MatmulExtraLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
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

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

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

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& extra,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool formatA, bool formatB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams  params;
        GetKernelInfo(mat1, mat2, outDType, transA, transB, formatA, formatB, tParams, params);

        if constexpr (IsBias) {
            TORCH_CHECK(extra.dim() == 1 && extra.size(0) == static_cast<int64_t>(params.n),
                        "bias must be a 1-D tensor of size N=", params.n,
                        ", got ", extra.dim(), "-D of size ",
                        extra.dim() > 0 ? extra.size(0) : 0);
            params.inputAddr.resize(3);
            params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(extra.storage().data()));
            // bias mode: allocate fresh output
            params.outputAddr.resize(1);
            auto output = GetOutputTensor(
                {params.m, params.n}, static_cast<torch::Dtype>(outDType));
            params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

            aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
            uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
            return output;
        } else {
            // addend mode: in-place — extra is both addend and output
            TORCH_CHECK(extra.dim() == 2 &&
                        extra.size(0) == static_cast<int64_t>(params.m) &&
                        extra.size(1) == static_cast<int64_t>(params.n),
                        "addend must have shape (", params.m, ", ", params.n, ")");
            TORCH_CHECK(extra.scalar_type() == outDType,
                        "addend dtype ", extra.scalar_type(), " must match outDType ", outDType);
            params.outputAddr.resize(1);
            params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(extra.storage().data()));

            aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
            uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
            return extra;
        }
    }
};

} // namespace CatlassKernelWrapper

#endif
