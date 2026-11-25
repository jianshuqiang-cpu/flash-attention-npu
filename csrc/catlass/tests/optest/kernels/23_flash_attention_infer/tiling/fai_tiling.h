/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EXAMPLES_FAI_SHAREDLIB_TILING_H
#define CATLASS_EXAMPLES_FAI_SHAREDLIB_TILING_H

#include <cstdint>
#include "fai_tiling_data.h"

namespace FAInferTiling {
const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM128 = 128;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t WORKSPACE_BLOCK_SIZE_DB = 131072;

enum class MaskType{
    NO_MASK = 0, 
    MASK_SPEC = 1, 
    MASK_CAUSUAL = 2
};

struct FAInfo {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int64_t *qSeqlenList{nullptr};
    int64_t *kvSeqlenList{nullptr};
    int64_t *qSeqlen{nullptr};
    MaskType maskType = MaskType::MASK_SPEC;
};
int32_t GetFATilingParam(const FAInfo &faInfo, uint32_t blockDim, FATilingData &faTilingData);
} // namespace FAInferTiling
#endif