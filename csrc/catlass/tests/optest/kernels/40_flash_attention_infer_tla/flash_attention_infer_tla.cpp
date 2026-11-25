/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <acl/acl.h>
#include <iostream>

#include "catlass_kernel_prebuilt.h"
#include "kernel_operator.h"
#include "tiling/fai_tiling.h"

#include "fai_kernel.cpp"



namespace CatlassKernel {

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        int32_t error = status;                                                              \
        if (error != 0) {                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

template <class DType>
void FAImpl(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams &params)
{
    int32_t batch = params.batch;
    int32_t qSeqlen = params.qSeqlen;
    int32_t kvSeqlen = params.kvSeqlen;
    int32_t numHeads = params.numHeads;
    int32_t kvHeads = params.kvHeads;
    int32_t embeddingSize = params.embeddingSize;
    int32_t blockSize = params.blockSize;
    int32_t maskType = params.maskType;
    int32_t maxKvSeqlen = kvSeqlen;
    int32_t numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);
    int32_t numTokens = params.qNtokens;

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint32_t tilingSize = sizeof(FATilingData);

    uint8_t *qSeqDevice = params.inputAddr.at(0);
    uint8_t *kvSeqDevice = params.inputAddr.at(1);
    uint8_t *qDevice = params.inputAddr.at(2);
    uint8_t *kDevice = params.inputAddr.at(3);
    uint8_t *vDevice = params.inputAddr.at(4);

    uint8_t *maskDevice = nullptr;
    if (maskType == 1) {
        maskDevice = params.inputAddr.at(5);
    }

    uint8_t *blockTableDevice = params.inputAddr.at(6);
    uint8_t *oDevice = params.outputAddr.at(0);

    uint64_t mm1OutSize = blockNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t smOnlineOutSize = blockNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(DType) * FAInferTiling::NUM3;
    uint64_t mm2OutSize = blockNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;
    uint64_t updateSize = blockNum * FAInferTiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * FAInferTiling::NUM3;

    uint8_t *sDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&sDevice), mm1OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *pDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&pDevice), smOnlineOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oTempDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&oTempDevice), mm2OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *oUpdateDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&oUpdateDevice), updateSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = blockNum;

    FAInferTiling::FAInfo faInfo;
    faInfo.numTokens = numTokens;
    faInfo.numHeads = numHeads;
    faInfo.embeddingSize = embeddingSize;
    faInfo.numBlocks = numBlocks;
    faInfo.blockSize = blockSize;
    faInfo.kvHeads = kvHeads;
    faInfo.batch = batch;
    faInfo.maskType = static_cast<FAInferTiling::MaskType>(maskType);

    uint8_t *qSeqHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&qSeqHost), seqArraySize));
    ACL_CHECK(aclrtMemcpy(qSeqHost, seqArraySize, qSeqDevice, seqArraySize, ACL_MEMCPY_DEVICE_TO_HOST));
    uint8_t *kvSeqHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&kvSeqHost), seqArraySize));
    ACL_CHECK(aclrtMemcpy(kvSeqHost, seqArraySize, kvSeqDevice, seqArraySize, ACL_MEMCPY_DEVICE_TO_HOST));

    faInfo.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faInfo.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);

    FATilingData faTilingData;
    FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, &faTilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t hardwareSyncAddr{0};
    ACL_CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void**>(&hardwareSyncAddr)));

    FAInferTla<DType><<<blockDim, nullptr, stream>>>(
        hardwareSyncAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, qSeqDevice, kvSeqDevice, sDevice,
        pDevice, oTempDevice, oUpdateDevice, tilingDevice);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTempDevice);
    aclrtFree(oUpdateDevice);
    aclrtFree(tilingDevice);
    aclrtFreeHost(qSeqHost);
    aclrtFreeHost(kvSeqHost);
    aclrtFreeHost(tilingHost);
}

void FlashAttentionInferTLA(uint32_t blockNum, aclrtStream stream, const FlashAttentionParams &params)
{
    if (params.dataType == ACL_FLOAT16) {
        FAImpl<half>(blockNum, stream, params);
    } else if (params.dataType == ACL_BF16) {
        FAImpl<bfloat16_t>(blockNum, stream, params);
    }
}

} // namespace CatlassKernel
