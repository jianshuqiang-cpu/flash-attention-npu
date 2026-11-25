/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <acl/acl.h>
#include <cmath>

#include "catlass_kernel_prebuilt.h"
#include "kernel_operator.h"
#include "tiling/fai_tiling.h"

#include "fai_kernel.h"

namespace CatlassKernel {

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        (void)error;                                                                         \
    } while (0)

template <class DType, bool enableMaskFlag, bool enablePaFlag>
void FAImplDispatch(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams &params)
{
    int32_t batch = params.batch;
    int32_t qSeqlen = params.qSeqlen;
    int32_t kvSeqlen = params.kvSeqlen;
    int32_t numHeads = params.numHeads;
    int32_t kvHeads = params.kvHeads;
    int32_t embeddingSize = params.embeddingSize;
    int32_t blockSize = params.blockSize;
    int32_t maskType = params.maskType;
    uint32_t tilingSize = sizeof(FATilingData);

    uint8_t *qSeqDevice = params.inputAddr.at(0);
    uint8_t *kvSeqDevice = params.inputAddr.at(1);
    uint8_t *qDevice = params.inputAddr.at(2);
    uint8_t *kDevice = params.inputAddr.at(3);
    uint8_t *vDevice = params.inputAddr.at(4);

    uint8_t *maskDevice = nullptr;
    if (maskType > 0) {
        maskDevice = params.inputAddr.at(5);
    }

    uint8_t *blockTableDevice = params.inputAddr.at(6);
    uint8_t *oDevice = params.outputAddr.at(0);

    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint32_t blockDim = blockNum;

    FAInferTiling::FAInfo faInfo;
    faInfo.batchSize = batch;
    faInfo.numOfHeads = numHeads;
    faInfo.numOfKVHeads = kvHeads;
    faInfo.seqSize = qSeqlen;
    faInfo.seqInnerSize = kvSeqlen;
    faInfo.headSize = embeddingSize;
    faInfo.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * faInfo.headSize));
    faInfo.blockSize = blockSize;
    faInfo.maxBlockNumPerBatch = (kvSeqlen + blockSize - 1) / blockSize;

    FATilingData faTilingData;
    FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &faTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    FAInferTla<DType, enableMaskFlag, enablePaFlag><<<blockDim, nullptr, stream>>>(
        qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice, tilingDevice);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclrtFree(tilingDevice);
}

template <class DType>
void FAImpl(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams &params)
{
    constexpr bool enablePa = true;
    if (params.maskType > 0 && enablePa) {
        FAImplDispatch<DType, true, true>(blockNum, stream, params);
    } else if (params.maskType > 0) {
        FAImplDispatch<DType, true, false>(blockNum, stream, params);
    } else if (enablePa) {
        FAImplDispatch<DType, false, true>(blockNum, stream, params);
    } else {
        FAImplDispatch<DType, false, false>(blockNum, stream, params);
    }
}

void Ascend950FlashAttentionInfer(uint32_t blockNum, aclrtStream stream, const FlashAttentionParams &params)
{
    if (params.dataType == ACL_FLOAT16) {
        FAImpl<half>(blockNum, stream, params);
    } else if (params.dataType == ACL_BF16) {
        FAImpl<bfloat16_t>(blockNum, stream, params);
    }
}

} // namespace CatlassKernel
