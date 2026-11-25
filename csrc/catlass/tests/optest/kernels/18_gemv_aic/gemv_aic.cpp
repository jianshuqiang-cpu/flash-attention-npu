#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void GemvAIC(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("gemv_aic", tParams);
    auto* entry = JitCompiler::instance().getKernel("gemv_aic_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
