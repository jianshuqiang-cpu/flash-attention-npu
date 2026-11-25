/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fai_tiling.h"

namespace FAInferTiling {
namespace {

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr int32_t SPARSE_MODE_RIGHT_DOWN = 2;
constexpr int32_t BLOCK_BASE_SIZE = 128;

template <typename T>
T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

void GetPreNextTokensLeftUp(
    FATilingData &tilingData,
    int64_t actualSeqLength,
    int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp,
    int64_t &nextTokensLeftUp)
{
    auto &baseParams = tilingData.inputParamsRegbase;
    int64_t preTokens = SPARSE_MODE_INT_MAX;
    int64_t nextTokens = SPARSE_MODE_INT_MAX;
    if (baseParams.attenMaskCompressMode == SPARSE_MODE_LEFT_UP) {
        preTokens = SPARSE_MODE_INT_MAX;
        nextTokens = 0;
    }
    if (baseParams.attenMaskCompressMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = preTokens;
        nextTokensLeftUp = nextTokens;
    }
}

void FixParamWithRowInvalid(
    int64_t &actualSeqLength,
    int64_t actualSeqLengthKV,
    int64_t &preTokensLeftUp,
    int64_t &nextTokensLeftUp)
{
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
        (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;

    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;
    actualSeqLength -= preTokensError;
}

int64_t GetCutBlockNums(
    int64_t blockSeqLengthKV,
    int64_t blockSeqLength,
    int64_t sInner,
    int64_t sOuter,
    int64_t token)
{
    if (sInner == 0 || sOuter == 0) {
        return 0;
    }
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    int64_t tmpInnerCutBlockNums = (innerCutBlockNums > 0) ?
        (innerCutBlockNums % tolerance + innerCutBlockNums) * (innerCutBlockNums / tolerance + 1) / 2 : 0;
    blockNums += tmpInnerCutBlockNums;
    int64_t tmpInnerCutBlockLeftNums = (innerCutBlockLeftNums > 0) ?
        (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) * (innerCutBlockLeftNums / tolerance + 1) / 2 : 0;
    blockNums -= tmpInnerCutBlockLeftNums;
    int64_t tmpInnerCutBlockDownNums = (innerCutBlockDownNums > 0) ?
        (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) * (innerCutBlockDownNums / tolerance + 1) / 2 : 0;
    blockNums -= tmpInnerCutBlockDownNums;
    return blockNums;
}

int64_t GetCalcBlockNumsOneHead(
    int64_t actualSeqLength,
    int64_t actualSeqLengthKV,
    int64_t sOuterSize,
    int64_t sInnerSize,
    int64_t preTokensLeftUp,
    int64_t nextTokensLeftUp,
    bool isAttenMaskUsed)
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        return innerBlockNums * outerBlockNums;
    }

    int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
    int64_t blockSeqLengthKV = innerBlockNums * sInnerSize;
    int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
    int64_t blockSeqLength = outerBlockNums * sOuterSize;
    int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    toCalcBlockNums -= GetCutBlockNums(
        blockSeqLengthKV, blockSeqLength, sInnerSize, sOuterSize, nextTokensLeftUp);
    toCalcBlockNums -= GetCutBlockNums(
        blockSeqLengthKV, blockSeqLength, sInnerSize, sOuterSize,
        blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
    return toCalcBlockNums;
}

int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums)
{
    if (sInnerIndexEnd < 0) {
        return 0;
    }
    if (sInnerIndexEnd < innerBlockNums) {
        return (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    }
    int64_t tmpSInnerBlockNums = sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0;
    return (sInnerIndexStart < 0) ? innerBlockNums : tmpSInnerBlockNums;
}

void ComputeSplitNBSeq(
    FATilingData &tilingData,
    uint32_t batchSize,
    const size_t tilingElementArrayLen,
    std::vector<int64_t> &actualSeqLengths,
    std::vector<int64_t> &actualSeqLengthsKV,
    int64_t sOuterSize,
    int64_t sInnerSize,
    double coreWightTarget,
    uint32_t &curCore)
{
    auto &baseParams = tilingData.inputParamsRegbase;
    std::vector<uint32_t> bnAxisStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> qSeqAxisStartIdx(tilingElementArrayLen, 0L);
    int64_t curWeight = 0;
    uint32_t lastHeadIdx = 0;
    uint32_t lastBatchIdx = 0;
    uint32_t lastQSeqOuterIdx = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (uint32_t headNum = 0; headNum < baseParams.qHeads; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(
                tilingData, actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx],
                preTokensLeftUp, nextTokensLeftUp);
            FixParamWithRowInvalid(
                actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx],
                preTokensLeftUp, nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLengths[batchIdx] + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthsKV[batchIdx] + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t diff = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWeight;
                int64_t sInnerIndexStart = -(preTokensLeftUp > 0 ?
                    (preTokensLeftUp + sInnerSize - 1) / sInnerSize : preTokensLeftUp / sInnerSize);
                int64_t sInnerIndexEnd = nextTokensLeftUp > 0 ?
                    (nextTokensLeftUp + sInnerSize - 1) / sInnerSize : nextTokensLeftUp / sInnerSize;
                int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                if (sInnerBlockNums - diff > diff &&
                    !(lastHeadIdx == 0 && lastBatchIdx == 0 && lastQSeqOuterIdx == 0)) {
                    curCore += 1;
                    bnAxisStartIdx[curCore] = batchIdx * baseParams.qHeads + headNum;
                    qSeqAxisStartIdx[curCore] = sOuterIndex;
                }
                lastHeadIdx = headNum + 1;
                lastBatchIdx = batchIdx + 1;
                lastQSeqOuterIdx = sOuterIndex + 1;
                curWeight += sInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }
    bnAxisStartIdx[curCore + 1] = batchSize * baseParams.qHeads;
    qSeqAxisStartIdx[curCore + 1] = static_cast<int64_t>(lastQSeqOuterIdx);
    std::copy(
        std::begin(bnAxisStartIdx), std::end(bnAxisStartIdx),
        std::begin(tilingData.multiCoreParamsRegbase.bnAxisStartIdx));
    std::copy(
        std::begin(qSeqAxisStartIdx), std::end(qSeqAxisStartIdx),
        std::begin(tilingData.multiCoreParamsRegbase.sparseStartIdx));
}

void FillInputParams(const FAInfo &faInfo, FATilingData &tilingData)
{
    auto &inputParams = tilingData.inputParamsRegbase;
    inputParams.batch = faInfo.batchSize;
    inputParams.qHeads = faInfo.numOfHeads;
    inputParams.kvHeads = faInfo.numOfKVHeads;
    inputParams.groupSize = faInfo.numOfHeads / faInfo.numOfKVHeads;
    inputParams.qSeqlen = faInfo.seqSize;
    inputParams.kvSeqlen = faInfo.seqInnerSize;
    inputParams.embed = faInfo.headSize;
    inputParams.scaleValue = faInfo.scaleValue;
    inputParams.attenMaskCompressMode = faInfo.maskType;
    inputParams.headNumRatio = static_cast<uint32_t>(faInfo.numOfHeads / faInfo.numOfKVHeads);
    inputParams.blockSize = faInfo.blockSize;
    inputParams.blockTableDim2 = faInfo.maxBlockNumPerBatch;
    inputParams.paBlockNumSum = faInfo.numBlocks;
    inputParams.attenMaskQSeqlen = static_cast<uint32_t>(faInfo.seqSize);
    inputParams.attenMaskKvSeqlen = static_cast<uint32_t>(faInfo.seqInnerSize);
}

void FillActualSeqLengths(
    const FAInfo &faInfo,
    FATilingData &tilingData,
    std::vector<int64_t> &actualSeqLengths,
    std::vector<int64_t> &actualSeqLengthsKV)
{
    auto &inputParams = tilingData.inputParamsRegbase;
    int64_t batchSize = inputParams.batch;
    bool isActualSeqLengthsNull = faInfo.actualSeqLengths == nullptr;
    bool isActualSeqLengthsKVNull = faInfo.actualSeqLengthsKV == nullptr;
    inputParams.isActualSeqLengthsNull = isActualSeqLengthsNull;
    inputParams.isActualSeqLengthsKVNull = isActualSeqLengthsKVNull;
    inputParams.actualSeqLengthsSize = static_cast<uint32_t>(isActualSeqLengthsNull ? batchSize : 0);
    inputParams.actualSeqLengthsKVSize = static_cast<uint32_t>(isActualSeqLengthsKVNull ? batchSize : 0);
    for (int64_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        actualSeqLengths[batchIdx] = isActualSeqLengthsNull ?
            inputParams.qSeqlen : faInfo.actualSeqLengths[batchIdx];
        actualSeqLengthsKV[batchIdx] = isActualSeqLengthsKVNull ?
            inputParams.kvSeqlen : faInfo.actualSeqLengthsKV[batchIdx];
    }
}

} // namespace

int32_t GetFATilingParam(const FAInfo &faInfo, uint32_t blockDim, FATilingData &faTilingData)
{
    FillInputParams(faInfo, faTilingData);

    auto &inputParams = faTilingData.inputParamsRegbase;
    int64_t batchSize = inputParams.batch;
    std::vector<int64_t> actualSeqLengths(batchSize);
    std::vector<int64_t> actualSeqLengthsKV(batchSize);
    FillActualSeqLengths(faInfo, faTilingData, actualSeqLengths, actualSeqLengthsKV);

    bool isAttenMaskUsed = faInfo.maskType != SPARSE_MODE_NO_MASK;
    int64_t totalBlockNumsOneHead = 0;
    constexpr int64_t sInnerSize = BLOCK_BASE_SIZE;
    constexpr int64_t sOuterSize = BLOCK_BASE_SIZE;
    for (int64_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[batchIdx];
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(
            faTilingData, actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx],
            preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(
            actualSeqLengthsTmp, actualSeqLengthsKV[batchIdx], preTokensLeftUp, nextTokensLeftUp);
        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(
            actualSeqLengthsTmp, actualSeqLengthsKV[batchIdx], sOuterSize, sInnerSize,
            preTokensLeftUp, nextTokensLeftUp, isAttenMaskUsed);
    }

    double coreWeightTarget = double(totalBlockNumsOneHead * inputParams.qHeads) / double(blockDim);
    int64_t qSeqlenOuterSize = (inputParams.qSeqlen + sOuterSize - 1) / sOuterSize;
    const size_t tilingElementArrayLen = MAX_CORE_NUM;
    uint32_t curIndx = 0;
    ComputeSplitNBSeq(
        faTilingData, batchSize, tilingElementArrayLen, actualSeqLengths, actualSeqLengthsKV,
        sOuterSize, sInnerSize, coreWeightTarget, curIndx);

    int64_t sInnerBlockNum = (inputParams.kvSeqlen + sInnerSize - 1) / sInnerSize;
    int64_t totalSize = (totalBlockNumsOneHead / sInnerBlockNum) * inputParams.qHeads;
    faTilingData.multiCoreParamsRegbase.qSeqlenOuterSize = qSeqlenOuterSize;
    faTilingData.multiCoreParamsRegbase.coreNum = static_cast<int32_t>(curIndx + 1);
    faTilingData.multiCoreParamsRegbase.totalSize = totalSize;
    faTilingData.multiCoreParamsRegbase.splitFactorSize =
        CeilDivision(totalSize, static_cast<int64_t>(curIndx + 1));
    faTilingData.multiCoreParamsRegbase.splitFactorTailSize = CalcTailSize(
        totalSize, faTilingData.multiCoreParamsRegbase.splitFactorSize);
    return 0;
}

} // namespace FAInferTiling
