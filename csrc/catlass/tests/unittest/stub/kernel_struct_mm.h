/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_STRUCT_MM_H
#define ASCENDC_STUB_KERNEL_STRUCT_MM_H

#include <cstdio>
#include <string>
#include <cstdint>

#include "kernel_struct_fixpipe.h"
#include "kernel_constants.h"
#include "kernel_fp_types.h"

using namespace AscendC;

namespace AscendC {

// ============================================================================
// FmatrixMode enum
// ============================================================================

enum class FmatrixMode : uint8_t {
    FMATRIX_LEFT = 0,
    FMATRIX_RIGHT = 1,
};

// ============================================================================
// MmadParams
// ============================================================================

struct MmadParams {
    MmadParams() = default;

    MmadParams(const uint16_t mIn, const uint16_t nIn, const uint16_t kIn, const uint8_t unitFlagIn,
        const bool cmatrixSourceIn, const bool cmatrixInitValIn)
        : m(mIn),
          n(nIn),
          k(kIn),
          unitFlag(unitFlagIn),
          cmatrixSource(cmatrixSourceIn),
          cmatrixInitVal(cmatrixInitValIn)
    {}

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    bool isBias = false;
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    bool disableGemv = false;
#endif

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "MmadParams(m=%u, n=%u, k=%u, unitFlag=%u, cmatrixInitVal=%s)",
                m, n, k, unitFlag, cmatrixInitVal ? "true" : "false");
        return std::string(buffer);
    }
};

// ============================================================================
// LoadData2DParams-Series
// ============================================================================

using LoadData2dParams = struct LoadData2DParams;
struct LoadData2DParams {
    LoadData2DParams() = default;

    LoadData2DParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn, const uint16_t srcStrideIn,
        const uint8_t sidIn, const uint16_t dstGapIn, const bool ifTransposeIn, const uint8_t addrModeIn)
        : startIndex(startIndexIn),
          repeatTimes(repeatTimesIn),
          srcStride(srcStrideIn),
          sid(sidIn),
          dstGap(dstGapIn),
          ifTranspose(ifTransposeIn),
          addrMode(addrModeIn)
    {}

    uint16_t startIndex = 0;
    uint16_t dstGap = 0;
    uint16_t srcStride = 0;
    bool ifTranspose = 0;
    uint8_t repeatTimes = 0;

    uint8_t sid = 0;
    uint8_t addrMode = 0;
    std::string toString() const 
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2DParams(startIndex=%u, dstGap=%d, srcStride=%d, ifTranspose=%d, repeatTimes=%d, sid=%d, addrMode=%d)",
                startIndex, dstGap, srcStride, ifTranspose, repeatTimes, sid, addrMode);
        return std::string(buffer);
    }
};

struct LoadData2DParamsV2 {
    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2DParamsV2(mPos=%u, kPos=%u, mStep=%u, kStep=%u)",
                mStartPosition, kStartPosition, mStep, kStep);
        return std::string(buffer);
    }
};

// ============================================================================
// LoadData2dTransposeParams-Series
// ============================================================================

struct LoadData2dTransposeParams {
    LoadData2dTransposeParams() = default;

    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint8_t addrMode = 0;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2dTransposeParams(startIndex=%u, repeatTimes=%u, srcStride=%u, dstGap=%u, dstFracGap=%u, addrMode=%u)",
                startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode);
        return std::string(buffer);
    }
};

struct LoadData2dTransposeParamsV2 {
    uint16_t startIndex = 0;
    uint8_t repeatTimes = 0;
    uint16_t srcStride = 0;
    uint16_t dstGap = 0;
    uint16_t dstFracGap = 0;
    uint16_t srcFracGap = 0;
    uint8_t addrMode = 0;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2dTransposeParamsV2(startIndex=%u, repeatTimes=%u, srcStride=%u, dstGap=%u, dstFracGap=%u, srcFracGap=%u, addrMode=%u)",
                startIndex, repeatTimes, srcStride, dstGap, dstFracGap, srcFracGap, addrMode);
        return std::string(buffer);
    }
};

// ============================================================================
// LoadData2DMxParams  (CAN 9.0.0 compatible field names)
// ============================================================================

#if (defined(__NPU_ARCH__) && __NPU_ARCH__==3510)
struct LoadData2DMxParams {
    LoadData2DMxParams() = default;

    uint16_t xStartPosition = 0;
    uint16_t yStartPosition = 0;
    uint8_t xStep = 0;
    uint8_t yStep = 0;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2DMxParams(xPos=%u, yPos=%u, xStep=%u, yStep=%u)",
                xStartPosition, yStartPosition, xStep, yStep);
        return std::string(buffer);
    }
};

#endif

// ============================================================================
// LoadData3DParamsV1
// ============================================================================

template <typename TYPE>
struct LoadData3DParamsV1 {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    using T = typename std::conditional_t<Std::is_one_of_v<TYPE, fp8_e5m2_t, fp8_e4m3fn_t, fp8_e8m0_t, hifloat8_t, fp4x2_e1m2_t, fp4x2_e2m1_t>, uint8_t, TYPE>;
#else
    using T = TYPE;
#endif

    LoadData3DParamsV1() = default;

    uint8_t padList[PAD_SIZE] = {0};
    uint8_t strideW = 0;
    uint8_t strideH = 0;
    uint8_t filterW = 0;
    uint8_t filterH = 0;
    uint8_t dilationFilterW = 0;
    uint8_t dilationFilterH = 0;
    uint8_t jumpStride = 0;
    uint8_t repeatMode = 0;
    uint8_t repeatTime = 0;
    uint8_t cSize = 0;
    T padValue = 0;
    uint8_t fetchFilterW = 0;
    uint8_t fetchFilterH = 0;
    uint16_t l1H = 0;
    uint16_t l1W = 0;
    uint16_t c1Index = 0;
    int16_t leftTopW = 0;
    int16_t leftTopH = 0;

    std::string toString() const {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "LoadData3DParamsV1(padList=%s, strideW=%u, strideH=%u, filterW=%u, filterH=%u, dilationFilterW=%u, dilationFilterH=%u, jumpStride=%u, repeatMode=%u, repeatTime=%u, cSize=%u, padValue=%u, fetchFilterW=%u, fetchFilterH=%u, l1H=%u, l1W=%u, c1Index=%u, leftTopW=%d, leftTopH=%d)",
                padList, strideW, strideH, filterW, filterH, dilationFilterW, dilationFilterH, jumpStride, repeatMode, repeatTime, cSize, padValue, fetchFilterW, fetchFilterH, l1H, l1W, c1Index, leftTopW, leftTopH);
        return std::string(buffer);
    }
};

// ============================================================================
// LoadData3DParamsV2
// ============================================================================

template <typename TYPE>
struct LoadData3DParamsV2 {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    using T = typename std::conditional_t<Std::is_one_of_v<TYPE, fp8_e5m2_t, fp8_e4m3fn_t, fp8_e8m0_t, hifloat8_t, fp4x2_e1m2_t, fp4x2_e2m1_t>, uint8_t, TYPE>;
#else
    using T = TYPE;
#endif

    uint8_t padList[PAD_SIZE] = {0};
    uint16_t l1H = 0;
    uint16_t l1W = 0;
    uint16_t channelSize = 0;
    uint16_t kExtension = 0;
    uint16_t mExtension = 0;
    uint16_t kStartPt = 0;
    uint16_t mStartPt = 0;

    uint8_t strideW = 1;
    uint8_t strideH = 1;
    uint8_t filterW = 1;
    uint8_t filterH = 1;
    uint8_t dilationFilterW = 1;
    uint8_t dilationFilterH = 1;
    bool enTranspose = false;
    bool enSmallK = false;
    T padValue = 0;
    bool filterSizeW = false;
    bool filterSizeH = false;
    bool fMatrixCtrl = false;

    std::string toString() const {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "LoadData3DParamsV2(padList=%s, l1H=%u, l1W=%u, channelSize=%u, kExtension=%u, mExtension=%u, kStartPt=%u, mStartPt=%u, strideW=%u, strideH=%u, filterW=%u, filterH=%u, dilationFilterW=%u, dilationFilterH=%u, enTranspose=%d, enSmallK=%d, padValue=%u, filterSizeW=%d, filterSizeH=%d, fMatrixCtrl=%d)",
                padList, l1H, l1W, channelSize, kExtension, mExtension, kStartPt, mStartPt, strideW, strideH, filterW, filterH, dilationFilterW, dilationFilterH, enTranspose, enSmallK, padValue, filterSizeW, filterSizeH, fMatrixCtrl);
        return std::string(buffer);
    }
};

// ============================================================================
// LoadData3DParamsV2Pro
// ============================================================================

struct LoadData3DParamsV2Pro {
    LoadData3DParamsV2Pro() = default;

    uint16_t channelSize = 0;
    bool enTranspose = false;
    bool enSmallK = false;
    bool filterSizeW = false;
    bool filterSizeH = false;
    bool fMatrixCtrl = false;
    uint64_t extConfig = 0;
    uint64_t filterConfig = 0X10101010101;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData3DParamsV2Pro(channelSize=%u, enTranspose=%d, enSmallK=%d, filterSizeW=%d, filterSizeH=%d, fMatrixCtrl=%d)",
                channelSize, enTranspose, enSmallK, filterSizeW, filterSizeH, fMatrixCtrl);
        return std::string(buffer);
    }
};

// ============================================================================
// InitConstValueParams  (templated)
// ============================================================================

template <typename T>
struct InitConstValueParams {
    InitConstValueParams() = default;

    uint16_t repeatTimes = 0;
    uint16_t blockNum = 0;
    uint16_t dstGap = 0;
    T initValue = 0;
};

// ============================================================================
// LoadDataRepeatParam
// ============================================================================

struct LoadDataRepeatParam {
    LoadDataRepeatParam() = default;

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint8_t reserved = 0;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

struct LoadDataRepeatParamWithStride {
    LoadDataRepeatParamWithStride() = default;

    uint16_t repeatStride = 0;
    uint8_t repeatTime = 1;
    uint8_t repeatMode = 0;
    uint16_t dstStride = 0;
};
#endif

// ============================================================================
// IsResetLoad3dConfig (kept from project — used as LoadData template param)
// ============================================================================

struct IsResetLoad3dConfig {
    bool isSetFMatrix = true;
    bool isSetPadding = true;
};
constexpr IsResetLoad3dConfig IS_RESER_LOAD3D_DEFAULT_CONFIG = {true, true};

// ============================================================================
// DataCopyParams (from kernel_struct_data_copy.h)
// ============================================================================

struct DataCopyParams {
    constexpr DataCopyParams() = default;

    DataCopyParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint16_t dstStrideIn)
        : blockCount(count),
          blockLen(len),
          srcStride(srcStrideIn),
          dstStride(dstStrideIn)
    {}

    uint16_t blockCount = 0;
    uint16_t blockLen = 0;

    union {
        uint16_t srcGap = 0;
        uint16_t srcStride; // will be deprecated, use srcGap instead
    };

    union {
        uint16_t dstGap = 0;
        uint16_t dstStride; // will be deprecated, use dstGap instead
    };

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "DataCopyParams(blockCount=%u, blockLen=%u, srcStride=%u, dstStride=%u)",
                blockCount, blockLen, srcStride, dstStride);
        return std::string(buffer);
    }
};

// ============================================================================
// DataCopyCO12DstParams (from kernel_struct_data_copy.h )
// ============================================================================

struct DataCopyCO12DstParams {
    DataCopyCO12DstParams() = default;

    DataCopyCO12DstParams(const uint16_t nSizeIn, const uint16_t mSizeIn, const uint32_t dstStrideIn,
        const uint16_t srcStrideIn, const QuantMode_t quantPreIn, const uint8_t reluPreIn, const bool channelSplitIn,
        const bool nz2ndEnIn)
        : nSize(nSizeIn),
          mSize(mSizeIn),
          dstStride(dstStrideIn),
          srcStride(srcStrideIn),
          quantPre(quantPreIn),
          reluPre(reluPreIn),
          channelSplit(channelSplitIn),
          nz2ndEn(nz2ndEnIn)
    {}

    uint8_t sid = 0;
    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint32_t dstStride = DEFAULT_DATA_COPY_STRIDE;
    uint16_t srcStride = DEFAULT_DATA_COPY_STRIDE;
    uint8_t unitFlag = 0;
    uint8_t clipReluPre = 0;
    uint8_t eltWiseOp = 0;
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint8_t reluPre = 0;
    bool channelSplit = false;
    bool nz2ndEn = false;
};

// ============================================================================
// Nd2NzParams (from CANN 9.0.0 kernel_struct_data_copy.h)
// ============================================================================

struct Nd2NzParams {
    Nd2NzParams() = default;

#if (!defined(__NPU_ARCH__) || __NPU_ARCH__ == 2201)

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcNdMatrixStride = 0;  // Longer stride representation
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;
#else
    uint16_t ndNum = 0;
    uint16_t nValue = 0;
    uint16_t dValue = 0;
    uint16_t srcNdMatrixStride = 0;
    uint16_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint16_t dstNzMatrixStride = 0;
#endif

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Nd2NzParams(ndNum=%u, nValue=%u, dValue=%u)",
                ndNum, nValue, dValue);
        return std::string(buffer);
    }
};

// ============================================================================
// Dn2NzParams (from CAN 9.0.0 kernel_struct_data_copy.h)
// ============================================================================

struct Dn2NzParams {
    Dn2NzParams() = default;

    uint16_t dnNum = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t srcDnMatrixStride = 0;
    uint64_t srcDValue = 0;
    uint16_t dstNzC0Stride = 0;
    uint16_t dstNzNStride = 0;
    uint32_t dstNzMatrixStride = 0;

    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Dn2NzParams(dnNum=%u, nValue=%u, dValue=%u)",
                dnNum, nValue, dValue);
        return std::string(buffer);
    }
};

}  // namespace AscendC

#endif  // ASCENDC_STUB_KERNEL_STRUCT_MM_H
