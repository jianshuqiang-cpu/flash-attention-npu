/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNITTEST_TILE_SHAPE_HPP
#define UNITTEST_TILE_SHAPE_HPP

namespace Catlass::Test::Helper {

////////////// TILE SHAPE
struct TestMatrixShape {
    uint32_t row;
    uint32_t col;
};

struct TestMatrixShapeWithCoord {
    uint32_t row;
    uint32_t col;
    uint32_t rowCoord;
    uint32_t colCoord;
    uint32_t dstRow;
    uint32_t dstCol;
};

struct TestCubeMatrixShapeWithUnitflag {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    bool initC;
    uint8_t unitFlag;
};

struct TestVectorShape {
    uint32_t blkLen;
    uint16_t blkCnt;
};

struct TestVectorShapeWithStride : public TestVectorShape {
    uint32_t srcStride;
    uint32_t dstStride;
};

struct TestMatrixShapeWithUnitflag : public TestMatrixShape {
    uint8_t unitFlag;             // if unitFlag is enabled (0x11 -- disabled when copyout, 0x10 -- always on), the fractal will not accumulate on L0C
    bool channelSplit;         // if channelSplit is enabled, it will slice the output (SPLIT_M/SPLIT_N)

    TestMatrixShapeWithUnitflag(uint32_t row, uint32_t col)
        : TestMatrixShape{row, col}, unitFlag(0), channelSplit(false) {}
    
    TestMatrixShapeWithUnitflag(uint32_t row, uint32_t col, uint8_t unitFlag)
        : TestMatrixShape{row, col}, unitFlag(unitFlag), channelSplit(false) {}
};

}

#endif // UNITTEST_TILE_SHAPE_HPP