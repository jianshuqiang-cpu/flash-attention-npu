#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void BasicMatmulPreloadZN(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto* entry = JitCompiler::instance().getKernel(
        "basic_matmul_preload_zN_impl.cpp", JitMacroGenerator<TParams>::generate("basic_matmul_preload_zN", tParams), JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
