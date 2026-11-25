#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void GemvAIV(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GemmParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("gemv_aiv", tParams);
    auto* entry = JitCompiler::instance().getKernel("gemv_aiv_impl.cpp", macros, JitKernelType::AIV);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
