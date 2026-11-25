#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void GroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("grouped_matmul_slice_m", tParams);
    macros["CATLASS_JIT_K_GT_N"] = (params.k > params.n) ? "1" : "0";
    macros["L2_CACHE_HINT"] = "1";
    auto* entry = JitCompiler::instance().getKernel("grouped_matmul_slice_m_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
