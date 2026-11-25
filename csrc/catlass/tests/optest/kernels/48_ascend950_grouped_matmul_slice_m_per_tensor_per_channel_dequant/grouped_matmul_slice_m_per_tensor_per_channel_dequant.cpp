#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Ascend950GroupedMatmulSliceMPerTensorPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const GroupedMatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate(
        "ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant", tParams);
    macros["L2_CACHE_HINT"] = "1";
    auto quantModeIt = tParams.element.find("QUANT_MODE");
    int quantMode = (quantModeIt != tParams.element.end()) ? static_cast<int>(quantModeIt->second) : 0;
    macros["CATLASS_JIT_QUANT_MODE"] = std::to_string(quantMode);
    auto* entry = JitCompiler::instance().getKernel(
        "grouped_matmul_slice_m_per_tensor_per_channel_dequant_impl.cpp", macros, JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
