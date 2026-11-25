#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Gemm(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("gemm", tParams);
    auto* entry = JitCompiler::instance().getKernel("gemm_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
