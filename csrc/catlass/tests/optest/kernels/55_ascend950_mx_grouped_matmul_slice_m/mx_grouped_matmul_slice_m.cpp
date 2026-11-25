#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Ascend950MxGroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_mx_grouped_matmul_slice_m", tParams);
    macros["L2_CACHE_HINT"] = "1";
    macros["MX_GMM_ENABLE_ASWT"] = tParams.flagOn("ASWT") ? "1" : "0";
    macros["MX_GMM_ENABLE_PRELOAD"] = tParams.flagOn("PRELOAD") ? "1" : "0";
    auto* entry = JitCompiler::instance().getKernel(
        "mx_grouped_matmul_slice_m_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
