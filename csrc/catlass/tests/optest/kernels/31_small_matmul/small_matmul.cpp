#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void SmallMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto* entry = JitCompiler::instance().getKernel(
        "small_matmul_impl.cpp", JitMacroGenerator<TParams>::generate("small_matmul", tParams), JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
