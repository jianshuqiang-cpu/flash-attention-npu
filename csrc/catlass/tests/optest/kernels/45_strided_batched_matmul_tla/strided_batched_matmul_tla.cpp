#include "catlass_kernel.h"
#include "common/optimized_macro_generator.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void StridedBatchedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const StridedBatchedMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("strided_batched_matmul_tla", tParams);
    ApplyOptMacros(
        macros, params.m, params.n, params.k, tParams.nz("A"), tParams.trans("A"), tParams.nz("B"), tParams.trans("B"));
    auto* entry = JitCompiler::instance().getKernel("strided_batched_matmul_tla_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
