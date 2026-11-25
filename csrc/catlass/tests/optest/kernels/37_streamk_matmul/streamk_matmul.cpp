#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void StreamkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto* entry = JitCompiler::instance().getKernel(
        "streamk_matmul_impl.cpp", JitMacroGenerator<TParams>::generate("streamk_matmul", tParams), JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
