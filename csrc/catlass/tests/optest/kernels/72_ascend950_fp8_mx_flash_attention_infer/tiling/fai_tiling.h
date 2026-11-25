/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_MXFP8_FAI_TILING_H
#define OPTEST_MXFP8_FAI_TILING_H

#include <cstdint>

#include "tiling_data_def.h"

namespace FAInferTiling {

constexpr int32_t SPARSE_MODE_NO_MASK = 0;
constexpr int32_t SPARSE_MODE_LEFT_UP = 1;

struct FAInfo {
    int64_t batchSize = 0;
    int64_t numOfHeads = 0;
    int64_t numOfKVHeads = 0;
    int64_t seqSize = 0;
    int64_t seqInnerSize = 0;
    int64_t headSize = 0;

    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    uint32_t maskType = SPARSE_MODE_NO_MASK;
    float scaleValue = 1.0f;
    int64_t *actualSeqLengths{nullptr};
    int64_t *actualSeqLengthsKV{nullptr};
};

int32_t GetFATilingParam(const FAInfo &faInfo, uint32_t blockDim, FATilingData &faTilingData);

} // namespace FAInferTiling

#endif // OPTEST_MXFP8_FAI_TILING_H
