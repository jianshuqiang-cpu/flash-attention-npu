#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void PaddingSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("padding_splitk_matmul", tParams);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = (params.m > params.n) ? "30" : "31";
    auto* entry = JitCompiler::instance().getKernel(
        "padding_splitk_matmul_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
