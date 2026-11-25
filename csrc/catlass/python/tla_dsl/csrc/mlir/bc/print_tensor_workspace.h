#pragma once

#include <cstdint>

namespace tla::print_tensor {

// "TP" identifies TLA tensor-print records. Bits 17:16 identify an optional
// AIV subblock (1 for subblock 0, 2 for subblock 1); the low 16 bits store the
// static call ID. Tag 0 is used by AIC records.
constexpr uint32_t kDescriptorNamespace = 0x54500000;

[aicore] __attribute__((always_inline)) inline uint32_t EncodeDescriptor(uint64_t call_id, uint32_t subblock_tag = 0)
{
    return kDescriptorNamespace | (subblock_tag << 16) | (static_cast<uint32_t>(call_id) & 0xffffU);
}

[aicore] __attribute__((always_inline)) inline void InitializeWorkspace(uint64_t print_workspace)
{
    auto* print_address = reinterpret_cast<__gm__ uint8_t*>(print_workspace);
    // C310 DumpTensor emits TLVs through the shared debug FIFO. This also
    // bypasses CANN 9.1.0-beta.3's no-op InitDump overload.
    g_sysPrintFifoSpace = print_address;
}

} // namespace tla::print_tensor
