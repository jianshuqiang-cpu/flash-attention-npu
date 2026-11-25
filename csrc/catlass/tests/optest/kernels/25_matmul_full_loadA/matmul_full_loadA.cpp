#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void MatmulFullLoadA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("matmul_full_loadA", tParams);
    macros["CATLASS_JIT_BLOCK_SCHEDULER"] = (params.m > 128) ? "10" : "30";
    auto* entry = JitCompiler::instance().getKernel("matmul_full_loadA_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
