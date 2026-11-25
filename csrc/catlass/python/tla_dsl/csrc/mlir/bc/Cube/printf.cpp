#include "../common.h"

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_i8(int8_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_i16(int16_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_u8(uint8_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%u", static_cast<uint32_t>(value));
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_u16(uint16_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%u", static_cast<uint32_t>(value));
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_u32(uint32_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%u", value);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_x_i32(int32_t value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("x=%d", value);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_v_f16(half value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("v=%f", static_cast<float>(value));
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_v_f32(float value, uint64_t printWorkspace)
{
    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    AscendC::printf("v=%f", value);
}

[aicore] __attribute__((always_inline)) void tla_printf_write_format_record(
    const char* format, uint64_t formatLen, uint64_t argsNum, const uint64_t* args, uint64_t printWorkspace)
{
    constexpr uint32_t kScalarRecordType = 1;
    constexpr uint32_t kRingBufferOffset = 88;
    constexpr uint32_t kRingBufferBytes = 1024 * 1024;
    constexpr uint32_t kDebugBlockLength = 1048704;
    constexpr uint32_t kPrintTlvBytes = 24;
    constexpr uint64_t kPrintFmtOffset = 8;
    constexpr uint64_t kPrintRecordAlignment = sizeof(uint64_t);
    constexpr uint64_t kMaxArgs = 8;

    if (format == nullptr || argsNum > kMaxArgs || (argsNum != 0 && args == nullptr)) {
        return;
    }
    uint64_t argsBytes = argsNum * sizeof(uint64_t);
    if (formatLen + 1 > kRingBufferBytes - kPrintTlvBytes - argsBytes) {
        return;
    }

    g_sysPrintFifoSpace = reinterpret_cast<__gm__ uint8_t*>(printWorkspace);
    uint32_t blockIdx = AscendC::GetBlockIdx();
    __gm__ uint8_t* record = g_sysPrintFifoSpace + static_cast<uint64_t>(blockIdx) * kDebugBlockLength;
    __gm__ uint8_t* ring = record + kRingBufferOffset;
    __gm__ uint8_t* writeInfo = ring + kRingBufferBytes;
    __gm__ uint64_t* writeOffset = reinterpret_cast<__gm__ uint64_t*>(writeInfo + 8);
    uint64_t offset = *writeOffset;
    uint64_t rawTotal = kPrintTlvBytes + argsBytes + formatLen + 1;
    uint64_t total = (rawTotal + kPrintRecordAlignment - 1) & ~(kPrintRecordAlignment - 1);
    if (offset > kRingBufferBytes || total > kRingBufferBytes - offset) {
        return;
    }

    __gm__ uint8_t* tlv = ring + offset;
    *reinterpret_cast<__gm__ uint32_t*>(tlv) = kScalarRecordType;
    *reinterpret_cast<__gm__ uint32_t*>(tlv + 4) = static_cast<uint32_t>(total - 8);
    *reinterpret_cast<__gm__ uint32_t*>(tlv + 8) = blockIdx;
    *reinterpret_cast<__gm__ uint32_t*>(tlv + 12) = 0;
    *reinterpret_cast<__gm__ uint64_t*>(tlv + 16) = kPrintFmtOffset + argsBytes;
    for (uint64_t i = 0; i < argsNum; ++i) {
        *reinterpret_cast<__gm__ uint64_t*>(tlv + kPrintTlvBytes + i * sizeof(uint64_t)) = args[i];
    }
    __gm__ uint8_t* formatBytes = tlv + kPrintTlvBytes + argsBytes;
    for (uint64_t i = 0; i < formatLen; ++i) {
        formatBytes[i] = static_cast<uint8_t>(format[i]);
    }
    formatBytes[formatLen] = 0;
    for (uint64_t i = rawTotal; i < total; ++i) {
        tlv[i] = 0;
    }
    *writeOffset = offset + total;
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_string(
    const char* format, uint64_t formatLen, uint64_t printWorkspace)
{
    tla_printf_write_format_record(format, formatLen, 0, nullptr, printWorkspace);
}

[aicore] __attribute__((always_inline)) void _mlir_ciface_tla_printf_format_values(
    const char* format, uint64_t formatLen, uint64_t argsNum, uint64_t arg0, uint64_t arg1, uint64_t arg2,
    uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7, uint64_t printWorkspace)
{
    uint64_t args[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
    tla_printf_write_format_record(format, formatLen, argsNum, args, printWorkspace);
}
#endif
}
