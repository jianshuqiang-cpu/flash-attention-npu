#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void PlanarComplexMatmul(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("planar_complex_matmul", tParams);
    macros["CATLASS_JIT_USE_FOUR_PASS"] = tParams.flagOn("USE_FOUR_PASS") ? "1" : "0";
    macros["CATLASS_JIT_NEGATE_A"] = tParams.flagOn("NEGATE_A") ? "1" : "0";
    macros["CATLASS_JIT_SWIZZLE_DIR"] = tParams.flagOn("SWIZZLE_DIR_0") ? "0" : "1";
    auto* entry = JitCompiler::instance().getKernel("planar_complex_matmul_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
