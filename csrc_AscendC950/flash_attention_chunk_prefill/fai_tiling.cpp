/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "fai_tilingdata.h"

namespace optiling{
    const uint32_t SIZE_OF_16BIT = 2;
    const uint32_t SIZE_OF_32BIT = 4;
    const uint32_t N_SPLIT_HELPER = 2;
    const uint32_t MAX_KV_STACK_LEN = 512;
    const uint32_t Q_TILE_CEIL = 128;
    const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    const uint32_t BASE_KV_SIZE = 128;
    const uint32_t PRELANCH_NUM = 3;

    enum class MaskType : uint32_t {
        NO_MASK = 0,
        MASK_SPEC = 1,
        SWA_MASK = 4
    };

    enum class DataType : uint32_t {
        FP16 = 0,
        BF16 = 1
    };

    struct FAInferContext {
        int32_t numTokens = 0;
        int32_t numHeads = 0;
        int32_t embeddingSize = 0;
        int32_t embeddingSizeV = 0;
        int32_t numBlocks = 0;
        int32_t blockSize = 0;
        int32_t kvHeads = 0;
        int32_t batch = 0;
        int32_t innerPrecise = 0;
        int64_t maxQSeqlen = 0;
        int64_t maxKvSeqlen = 0;
        int64_t preToken = 0;
        int64_t nextToken = 0;
        int32_t sparseMode = 0;
        uint32_t globalWindowSize = 4;
        uint32_t localWindowSize = 0;
        std::string cacheLayout = "nd";
        uint32_t maxNumBlocksPerBatch = 0;
        const int64_t *qSeqlenList{nullptr};
        const int64_t *kvSeqlenList{nullptr};
        float scaleValue = 0.0;
        size_t* workspaces{nullptr};
        MaskType maskType = MaskType::MASK_SPEC;
        DataType dataType = DataType::FP16;
        bool pagedCacheFlag = false;
        bool lseFlag = false;
        bool isTilingSink = false;
        bool learnableSinkFlag = false;
        bool flashDecodeFlag = false;
        bool kvcacheNzFlag = false;
        std::string layout;
        bool pagedShapeFlag = false;
    };

    struct BatchParams {
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint32_t curQNBlockTile;
        uint32_t qNBlockNumPerGroup;
        uint32_t curQNBlockNum;
        uint32_t curQSBlockTile;
        uint32_t curQSBlockNum;
        uint32_t curKSBlockTile;
        uint32_t curKSBlockNum;
    };

    class FAInferTiling {
        public:
            FAInferTiling() = default;
            explicit FAInferTiling(const FAInferContext &faInfo);
            void DoTiling(FAInferTilingData &tilingdata);
            void SetCoreNum(uint32_t blockNum) {
                this->blockNum_ = blockNum;
            }
            uint32_t GetCoreNum() {
                return this->blockNum_; 
            }
            uint64_t GetTilingKey();
        private:
            void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
            void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
            uint32_t GetQSBlockTile(int64_t kvSeqlen);
            uint32_t GetKSBlockTile(int64_t kvSeqlen);
            uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
            void FillBasicTilingData(FAInferTilingData &faTilingData);
            void splitBN2S1GS2(FAInferTilingData &faTilingData);
            BatchParams getBatchParams(uint32_t bIdx, uint32_t groupSize);
            void fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize, uint64_t perCoreTaskNum);
            void fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize);
        private:
            FAInferContext faInfo_;
            uint32_t blockNum_;
    };

    FAInferTiling::FAInferTiling(const FAInferContext &faInfo): faInfo_(faInfo) {}

    uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qRowNumCeil = Q_TILE_CEIL;
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }
    uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
    {
        uint32_t kSBlockTile = 128;
        return kSBlockTile;
    }

    void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
    {
        faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
        faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
        faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

        faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
        faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
        faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
        if (faInfo_.pagedCacheFlag) {
            faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
        } else {
            faTilingData.set_blockSize(BASE_KV_SIZE);
        }
        faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
        faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
        faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
        faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
        faTilingData.set_scaleValue(faInfo_.scaleValue);
        faTilingData.set_sparseMode(faInfo_.sparseMode);
        faTilingData.set_globalWindowSize(faInfo_.globalWindowSize);
        faTilingData.set_localWindowSize(faInfo_.localWindowSize);
        faTilingData.set_cacheLayout(faInfo_.cacheLayout);
        faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
        faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
        
        auto mm1L1TileM_ = 128;
        auto mm1L1TileKLeft_ = 192;
        auto qL1BufNum_ = 1;
        // K矩阵开启2buf，D按128分割，S2按256分割
        auto mm1L1TileN_ = 256;
        auto mm1L1TileKRight_ = 192;
        auto kL1BufNum_ = 2;
        // V矩阵开启db，D按128分割，kvBaseTile_不分割，指令同样提前于核间同步下发
        // 如果kvBaseTile_进一步增大，考虑关闭db，使得kvBaseTile_不分割
        auto mm2L1TileN_ = 128;
        auto mm2L1TileKLeft_ = 256;
        auto vL1BufNum_ = 2;
        // P矩阵在950上会常驻L1，由于基块的prelaunch为2，因此最好有3 buf，以免基块间流水阻塞
        auto mm2L1TileM_ = 128;
        auto mm2L1TileKRight_ = 128;
        auto pL1BufNum_ = 3;
        faTilingData.set_innerPrec(0);
        faTilingData.set_actSeqAval(0);
        faTilingData.set_qBaseTile(128);
        faTilingData.set_kvBaseTile(128);
        faTilingData.set_mm1L1TileM(mm1L1TileM_);
        faTilingData.set_mm1L1TileN(mm1L1TileN_);
        faTilingData.set_mm1L1TileKLeft(mm1L1TileKLeft_);
        faTilingData.set_mm1L1TileKRight(mm1L1TileKRight_);
        faTilingData.set_mm2L1TileM(mm2L1TileM_);
        faTilingData.set_mm2L1TileN(mm2L1TileN_);
        faTilingData.set_mm2L1TileKLeft(mm2L1TileKLeft_);
        faTilingData.set_mm2L1TileKRight(mm2L1TileKRight_);
        faTilingData.set_qL1BufNum(qL1BufNum_);
        faTilingData.set_kL1BufNum(kL1BufNum_);
        faTilingData.set_vL1BufNum(vL1BufNum_);
        faTilingData.set_pL1BufNum(pL1BufNum_);

    }

    uint64_t FAInferTiling::GetTilingKey() 
    {
        constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
        constexpr uint64_t PAGED_CACHE_KEY = 10000000;
        constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
         constexpr uint64_t COMP_SWA_MASK_KEY = 5;
        constexpr uint64_t LAYOUTQ_BSND_KEY = 100000;
        constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
        constexpr uint64_t DTYPE_FP16_KEY = 100;
        constexpr uint64_t KVCACHE_NZ_KEY = 10;
        constexpr uint64_t DTYPE_BF16_KEY = 200;
        constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
        constexpr uint64_t INNER_LOW_PREC_KEY = 10000;
        constexpr uint64_t LEARNABLE_SINK_KEY = 100000000;
        constexpr uint64_t FLASH_DECODE_KEY = 100000000000000000;
        uint64_t tilingKey = SPLIT_FUSE_BASE_KEY;
        std::cout << "faInfo_.pagedCacheFlag:" << faInfo_.pagedCacheFlag << std::endl;
        std::cout << "faInfo_.maskType:" << (uint32_t)faInfo_.maskType << std::endl;
        std::cout << "faInfo_.layout:" << faInfo_.layout << std::endl;
        if (faInfo_.pagedCacheFlag) {
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.pagedShapeFlag) { // 20000000 表示 BnNBsD
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.maskType == MaskType::MASK_SPEC) {
            tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
        } else if (faInfo_.maskType == MaskType::SWA_MASK ) {
            tilingKey += static_cast<uint64_t>(COMP_SWA_MASK_KEY);
        }
        if (faInfo_.layout == "TND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_TND_KEY);
        } else if (faInfo_.layout == "BSND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_BSND_KEY);
        }
        if (faInfo_.dataType == DataType::FP16) {
            std::cout << "faInfo_.dataType:" << "fp16" << std::endl;
            tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
        } else if (faInfo_.dataType == DataType::BF16) {
            std::cout << "faInfo_.dataType:" << "bf16" << std::endl;
            tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
        }
        if (faInfo_.lseFlag) {
            tilingKey += static_cast<uint64_t>(LSE_OUT_ONLY_KEY);
        }
        if (faInfo_.learnableSinkFlag) {
            tilingKey += static_cast<uint64_t>(LEARNABLE_SINK_KEY);
        }
        if (faInfo_.innerPrecise == 1) {
            tilingKey += static_cast<uint64_t>(INNER_LOW_PREC_KEY);
        }
        if ((faInfo_.pagedCacheFlag) && !(faInfo_.maskType == MaskType::SWA_MASK) && !faInfo_.lseFlag && !faInfo_.learnableSinkFlag && !(faInfo_.innerPrecise == 1) && faInfo_.flashDecodeFlag) {
            tilingKey += static_cast<uint64_t>(FLASH_DECODE_KEY);
        }
        return tilingKey;
    }

    void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
    {
        uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_16BIT * PRELANCH_NUM;
        uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        
        uint64_t splitLseTotalSize = 0;
        uint64_t splitOTotalSize = 0;
        if (faInfo_.isTilingSink) {
            splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
            uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
            splitOTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT * faInfo_.numHeads;
            faTilingData.set_splitLseTotalSize(splitLseTotalSize);
            faTilingData.set_splitOTotalSize(splitOTotalSize);
            faTilingData.set_needCoreNum(blockNum_);
        } else {
            splitLseTotalSize = faTilingData.get_splitLseTotalSize();
            splitOTotalSize = faTilingData.get_splitOTotalSize();
        }
        uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize + splitLseTotalSize + splitOTotalSize;
        faTilingData.set_mm1OutSize(mm1OutSize);
        faTilingData.set_smOnlineOutSize(smOnlineOutSize);
        faTilingData.set_mm2OutSize(mm2OutSize);
        faTilingData.set_UpdateSize(UpdateSize);
        faTilingData.set_workSpaceSize(workSpaceSize);
    }

    void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
    {
        uint32_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
            uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
            if (faInfo_.layout == "TND") {
                uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx);
                qSeqlen = *(faInfo_.qSeqlenList + batchIdx + 1) - prevQSeqlenSum;
                if (!faInfo_.pagedCacheFlag) {
                    uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx);
                    kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx + 1) - prevKvSeqlenSum;
                }
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
            uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
            uint32_t curTaskNum = faInfo_.numHeads * curQSBlockNum;
            if (batchIdx == 0) {
                faTilingData.set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        faTilingData.set_totalTaskNum(totalTaskNum);
    }
    BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize) {
        BatchParams p;
        p.qSeqlen = *(faInfo_.qSeqlenList + bIdx);
        p.kvSeqlen = *(faInfo_.kvSeqlenList + bIdx);
        if (bIdx > 0 && faInfo_.layout == "TND") {
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + bIdx - 1);
            p.qSeqlen = p.qSeqlen - prevQSeqlenSum;
            if (!faInfo_.pagedCacheFlag) {
                uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + bIdx - 1);
                p.kvSeqlen = p.kvSeqlen - prevKvSeqlenSum;
            }
        }
        p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
        p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
        p.curQNBlockNum = p.qNBlockNumPerGroup * faInfo_.kvHeads;
        p.curQSBlockTile = GetQSBlockTile(p.kvSeqlen);
        p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
        p.curKSBlockTile = GetKSBlockTile(p.kvSeqlen); 
        p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
        return p;
    }

    void FAInferTiling::fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize, uint64_t perCoreTaskNum) {
        int32_t nowBIdx = 0;
        int32_t nowN1Idx = 0;
        int32_t nowS1Idx = 0;
        int32_t nowS2Idx = 0;
        
        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            faTilingData.coreInfo[coreIdx].startBIdx = 0;
            faTilingData.coreInfo[coreIdx].startN1Idx = 0;
            faTilingData.coreInfo[coreIdx].startS1Idx = 0;
            faTilingData.coreInfo[coreIdx].startS2Idx = 0;
            faTilingData.coreInfo[coreIdx].endBIdx = 0;
            faTilingData.coreInfo[coreIdx].endN1Idx = 0;
            faTilingData.coreInfo[coreIdx].endS1Idx = 0;
            faTilingData.coreInfo[coreIdx].endS2Idx = 0;
        }

        auto finishBatch = [&](uint32_t coreIdx) {
            BatchParams p = getBatchParams(faInfo_.batch - 1, groupSize);
            faTilingData.coreInfo[coreIdx].endBIdx = faInfo_.batch - 1;
            faTilingData.coreInfo[coreIdx].endN1Idx = p.curQNBlockNum - 1;
            faTilingData.coreInfo[coreIdx].endS1Idx = p.curQSBlockNum - 1;
            faTilingData.coreInfo[coreIdx].endS2Idx = p.curKSBlockNum;
            faTilingData.set_needCoreNum(coreIdx + 1);
        };

        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            int32_t resTaskNum = perCoreTaskNum;
            faTilingData.coreInfo[coreIdx].startBIdx = nowBIdx;
            faTilingData.coreInfo[coreIdx].startN1Idx = nowN1Idx;
            faTilingData.coreInfo[coreIdx].startS1Idx = nowS1Idx;
            faTilingData.coreInfo[coreIdx].startS2Idx = nowS2Idx;  

            BatchParams p = getBatchParams(nowBIdx, groupSize);

            auto advanceCounters = [&]() {
                if (nowS2Idx == p.curKSBlockNum) { nowS1Idx++; nowS2Idx = 0; }
                if (nowS1Idx == p.curQSBlockNum) { nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0; }
                if (nowN1Idx == p.curQNBlockNum) { nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0; }
            };

            while (nowS2Idx < p.curKSBlockNum && resTaskNum > 0) {
                p = getBatchParams(nowBIdx, groupSize); 
                uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
                uint32_t remainingKV = (nowS2Idx < p.curKSBlockNum - 1) ? p.curKSBlockTile : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
                uint64_t singleS2Task = remainingQ * remainingKV;
                resTaskNum -= singleS2Task;
                nowS2Idx += 1;
            }

            if (resTaskNum <= 0) {
                faTilingData.coreInfo[coreIdx].endBIdx = nowBIdx;
                faTilingData.coreInfo[coreIdx].endN1Idx = nowN1Idx;
                faTilingData.coreInfo[coreIdx].endS1Idx = nowS1Idx;
                faTilingData.coreInfo[coreIdx].endS2Idx = nowS2Idx;
            }
            
            advanceCounters();
            if (nowBIdx < faInfo_.batch && resTaskNum <= 0) continue;
            if (nowBIdx == faInfo_.batch) { finishBatch(coreIdx); break; }

            while (nowBIdx < faInfo_.batch && resTaskNum > 0) {
                p = getBatchParams(nowBIdx, groupSize);
                uint32_t remainingQ = p.qSeqlen * (faInfo_.numHeads - p.curQNBlockTile * nowN1Idx) - nowS1Idx * p.curQSBlockTile;
                uint32_t remainingKV = p.kvSeqlen;
                uint32_t remainingInBatch = remainingQ * remainingKV;

                if (resTaskNum >= remainingInBatch) {
                    resTaskNum -= remainingInBatch;
                    nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0;
                } else {
                    break;
                }
            }

            if (nowBIdx == faInfo_.batch) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            while (nowN1Idx < p.curQNBlockNum && resTaskNum > 0) {
                uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
                uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
                if (resTaskNum >= remainingInN1) {
                    resTaskNum -= remainingInN1;
                    nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0;
                } else {
                    break;
                }
            }
            
            advanceCounters();
            if (nowBIdx == faInfo_.batch) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            while (nowS1Idx < p.curQSBlockNum && resTaskNum > 0) {
                uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
                uint64_t remainingInS1 = remainingQ * p.kvSeqlen;
                if (resTaskNum >= remainingInS1) {
                    resTaskNum -= remainingInS1;
                    nowS1Idx++; nowS2Idx = 0;
                } else {
                    break;
                }
            }

            advanceCounters();
            if (nowBIdx == faInfo_.batch) { finishBatch(coreIdx); break; }
            p = getBatchParams(nowBIdx, groupSize);

            while (nowS2Idx < p.curKSBlockNum && resTaskNum > 0) {
                uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile : (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
                uint32_t remainingKV = (nowS2Idx < p.curKSBlockNum - 1) ? p.curKSBlockTile : (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
                uint64_t singleS2Task = remainingQ * remainingKV;
                resTaskNum -= singleS2Task;
                nowS2Idx += 1;
            }

            if (nowBIdx == faInfo_.batch) { finishBatch(coreIdx); break; }
            
            faTilingData.coreInfo[coreIdx].endBIdx = nowBIdx;
            faTilingData.coreInfo[coreIdx].endN1Idx = nowN1Idx;
            faTilingData.coreInfo[coreIdx].endS1Idx = nowS1Idx;
            faTilingData.coreInfo[coreIdx].endS2Idx = nowS2Idx;

            advanceCounters();
        }
    }

    void FAInferTiling::fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize) {
        for (uint32_t splitIdx = 0; splitIdx < blockNum_ + 1; splitIdx++) {
            faTilingData.splitInfo[splitIdx].batchIdx = 0;
            faTilingData.splitInfo[splitIdx].headStartIdx = 0;
            faTilingData.splitInfo[splitIdx].headEndIdx = 0;
            faTilingData.splitInfo[splitIdx].qStartIdx = 0;
            faTilingData.splitInfo[splitIdx].qEndIdx = 0;
            faTilingData.splitInfo[splitIdx].splitNum = 0;
            faTilingData.splitInfo[splitIdx].lseTaskOffset = 0;
            faTilingData.splitInfo[splitIdx].oTaskOffset = 0;
        }

        int64_t currentLseTaskOffset = 0;
        int64_t currentOTaskOffset = 0;
        int32_t splitIdx = -1;
        int32_t prevBIdx = -1;
        int32_t prevN1Idx = -1;
        int32_t prevS1Idx = -1;

        for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
            int32_t startBIdx = faTilingData.coreInfo[coreIdx].startBIdx;
            int32_t startN1Idx = faTilingData.coreInfo[coreIdx].startN1Idx;
            int32_t startS1Idx = faTilingData.coreInfo[coreIdx].startS1Idx;
            int32_t startS2Idx = faTilingData.coreInfo[coreIdx].startS2Idx;
            int32_t endBIdx = faTilingData.coreInfo[coreIdx].endBIdx;
            int32_t endN1Idx = faTilingData.coreInfo[coreIdx].endN1Idx;
            int32_t endS1Idx = faTilingData.coreInfo[coreIdx].endS1Idx;
            int32_t endS2Idx = faTilingData.coreInfo[coreIdx].endS2Idx;

            faTilingData.coreInfo[coreIdx].firstSplitKVTaskLseOffset = 0;
            faTilingData.coreInfo[coreIdx].firstSplitKVTaskOOffset = 0;

            bool foundFirstSplitKV = false;
            for (int BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
                BatchParams p = getBatchParams(BIdx, groupSize);
                
                int curStartN1 = (BIdx == startBIdx) ? startN1Idx : 0;
                int curEndN1 = (BIdx == endBIdx) ? endN1Idx : p.curQNBlockNum - 1;
                
                for (int N1Idx = curStartN1; N1Idx <= curEndN1; N1Idx++) {
                    int curStartS1 = (BIdx == startBIdx && N1Idx == startN1Idx) ? startS1Idx : 0;
                    int curEndS1 = (BIdx == endBIdx && N1Idx == endN1Idx) ? endS1Idx : p.curQSBlockNum - 1;

                    for (int S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                        int curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx && S1Idx == startS1Idx) ? startS2Idx : 0;
                        int curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx && S1Idx == endS1Idx) ? endS2Idx : p.curKSBlockNum;

                        int coveredS2 = curEndS2 - curStartS2;
                        bool isSplitKV = (coveredS2 > 0 && coveredS2 < p.curKSBlockNum);

                        int64_t tmpLseOffset = currentLseTaskOffset;
                        int64_t tmpOOffset = currentOTaskOffset;

                        uint32_t N1IdxPerGroup = N1Idx % p.qNBlockNumPerGroup;
                        uint32_t kvHeadIdx = N1Idx / p.qNBlockNumPerGroup;
                        uint32_t currentHeadStart = kvHeadIdx * groupSize + N1IdxPerGroup * p.curQNBlockTile;
                        uint32_t currentHeadEnd = std::min(currentHeadStart + p.curQNBlockTile, (kvHeadIdx + 1) * groupSize);

                        uint32_t currentQStart = S1Idx * p.curQSBlockTile;
                        uint32_t currentQEnd = std::min(currentQStart + p.curQSBlockTile, p.qSeqlen);

                        uint32_t headLen = currentHeadEnd - currentHeadStart;
                        uint32_t qLen = currentQEnd - currentQStart;

                        if (isSplitKV) {
                            if (BIdx != prevBIdx || N1Idx != prevN1Idx || S1Idx != prevS1Idx) {
                                splitIdx++;
                                if (splitIdx < blockNum_ + 1) {
                                    faTilingData.splitInfo[splitIdx].batchIdx = BIdx;
                                    faTilingData.splitInfo[splitIdx].splitNum = 0;

                                    faTilingData.splitInfo[splitIdx].headStartIdx = currentHeadStart;
                                    faTilingData.splitInfo[splitIdx].headEndIdx = currentHeadEnd;

                                    faTilingData.splitInfo[splitIdx].qStartIdx = currentQStart;
                                    faTilingData.splitInfo[splitIdx].qEndIdx = currentQEnd;
                                    
                                    faTilingData.splitInfo[splitIdx].lseTaskOffset = currentLseTaskOffset;
                                    faTilingData.splitInfo[splitIdx].oTaskOffset = currentOTaskOffset;
                                }
                                prevBIdx = BIdx; 
                                prevN1Idx = N1Idx; 
                                prevS1Idx = S1Idx;
                            }
                            if (splitIdx >= 0 && splitIdx < blockNum_ + 1) {
                                faTilingData.splitInfo[splitIdx].splitNum++;
                                currentLseTaskOffset += (int64_t)headLen * qLen;
                                currentOTaskOffset += (int64_t)headLen * qLen * faInfo_.embeddingSizeV;
                            }

                            if (!foundFirstSplitKV) {
                                foundFirstSplitKV = true;
                                faTilingData.coreInfo[coreIdx].firstSplitKVTaskLseOffset = tmpLseOffset;
                                faTilingData.coreInfo[coreIdx].firstSplitKVTaskOOffset = tmpOOffset;
                            }
                        }
                    }
                }
            }
        }

        uint32_t actualSplitNum = (splitIdx + 1 > (int32_t)blockNum_) ? blockNum_ : (splitIdx + 1);
        faTilingData.set_totalSplitNodeNum(actualSplitNum);
        faTilingData.set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
        faTilingData.set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
    }

    void FAInferTiling::splitBN2S1GS2(FAInferTilingData &faTilingData) {
        uint64_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;

        // 计算总任务数
        for (uint32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            BatchParams p = getBatchParams(batchIdx, groupSize);
            totalTaskNum += faInfo_.numHeads * p.qSeqlen * p.kvSeqlen;
        }
        uint64_t perCoreTaskNum = (totalTaskNum + blockNum_ - 1) / blockNum_;
        fillCoreInfoForFlashDecode(faTilingData, groupSize, perCoreTaskNum);
        fillSplitInfoForFlashDecode(faTilingData, groupSize);
    } 

    void FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
    {
        FillBasicTilingData(tilingdata);
        if (!faInfo_.isTilingSink) {
            FillSplitCoreTilingData(tilingdata);
            if (faInfo_.flashDecodeFlag) {
                splitBN2S1GS2(tilingdata);
            }
        }
        FillWorkSpaceTilingData(tilingdata);
    }
}