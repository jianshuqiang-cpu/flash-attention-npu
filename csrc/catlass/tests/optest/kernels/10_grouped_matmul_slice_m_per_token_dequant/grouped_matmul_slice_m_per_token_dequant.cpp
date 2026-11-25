#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void GroupedMatmulSliceMPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("grouped_matmul_slice_m_per_token_dequant", tParams);
    macros["L2_CACHE_HINT"] = "1";
    auto* entry = JitCompiler::instance().getKernel(
        "grouped_matmul_slice_m_per_token_dequant_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
