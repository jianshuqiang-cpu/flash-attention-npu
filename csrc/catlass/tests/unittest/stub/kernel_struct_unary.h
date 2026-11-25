/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_STRUCT_UNARY_H
#define ASCENDC_STUB_KERNEL_STRUCT_UNARY_H

#include <cstdio>
#include <string>
#include <cstdint>

namespace AscendC {

struct UnaryRepeatParams {
    uint32_t blockNumber = 1;
    uint16_t dstBlkStride = 1;
    uint16_t srcBlkStride = 1;
    uint8_t dstRepStride = 1;
    uint8_t srcRepStride = 1;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
    bool halfBlock = false;

    std::string toString() const
    {
        char buffer[256];
        snprintf(
            buffer, sizeof(buffer),
            "UnaryRepeatParams(dstBlkStride=%u, srcBlkStride=%u, dstRepStride=%u, srcRepStride=%u)", dstBlkStride,
            srcBlkStride, dstRepStride, srcRepStride);
        return std::string(buffer);
    }
};

struct BinaryRepeatParams {
    uint32_t blockNumber = 1;
    uint16_t dstBlkStride = 1;
    uint16_t src0BlkStride = 1;
    uint16_t src1BlkStride = 1;
    uint8_t dstRepStride = 1;
    uint8_t src0RepStride = 1;
    uint8_t src1RepStride = 1;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;

    std::string toString() const
    {
        char buffer[256];
        snprintf(
            buffer, sizeof(buffer), "BinaryRepeatParams(dstBlkStride=%u, src0BlkStride=%u, src1BlkStride=%u)",
            dstBlkStride, src0BlkStride, src1BlkStride);
        return std::string(buffer);
    }
};

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_STRUCT_UNARY_H