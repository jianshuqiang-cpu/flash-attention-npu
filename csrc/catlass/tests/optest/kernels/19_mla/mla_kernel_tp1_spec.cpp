/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "kernel_common.hpp"

using namespace Catlass;
/*
This example demonstrates how to compute mla.
*/
template <class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax, class EpilogueMLARescaleO,
          class EpilogueMLAFDRescaleO>
class MLAKernelTp1Spec {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = half;

    using ElementO = typename EpilogueMLARescaleO::ElementOutput;
    using LayoutO = typename EpilogueMLARescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueMLARescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueMLARescaleO::LayoutInput;

    using ElementUpdate = typename EpilogueMLARescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueMLARescaleO::LayoutUpdate;

    static constexpr uint32_t KV_SPLIT_MAX = EpilogueMLAFDRescaleO::KV_SPLIT_MAX;
    static constexpr uint32_t HEADS_PROCESS_MAX = EpilogueMLAFDRescaleO::HEADS_PROCESS_MAX;
    static constexpr uint32_t COMPUTE_ELE_NUM = EpilogueMLAFDRescaleO::COMPUTE_ELE_NUM;

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR q;
        GM_ADDR qRope;
        GM_ADDR k;
        GM_ADDR kRope;
        GM_ADDR blockTables;
        GM_ADDR o;
        GM_ADDR s;
        GM_ADDR p;
        GM_ADDR oTmp;
        GM_ADDR oUpdate;
        GM_ADDR oCoreTmp;
        GM_ADDR l;
        GM_ADDR tiling;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR q_, GM_ADDR qRope_, GM_ADDR k_, GM_ADDR kRope_, GM_ADDR blockTables_, GM_ADDR o_, GM_ADDR s_,
               GM_ADDR p_, GM_ADDR oTmp_, GM_ADDR oUpdate_, GM_ADDR oCoreTmp_, GM_ADDR l_, GM_ADDR tiling_)
            : q(q_), qRope(qRope_), k(k_), kRope(kRope_), blockTables(blockTables_), o(o_), s(s_), p(p_), oTmp(oTmp_),
              oUpdate(oUpdate_), oCoreTmp(oCoreTmp_), l(l_), tiling(tiling_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    MLAKernelTp1Spec() {}

    CATLASS_DEVICE void operator()(Params const &params)
    {
#if defined(__DAV_CUBE__)
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#elif defined(__DAV_VEC__)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
#endif

        // Get the memory offset address of the input on Global Memory
#if defined(__DAV_CUBE__)
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementQ> gQRope;
        gQRope.SetGlobalBuffer((__gm__ ElementQ *)params.qRope);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gKRope;
        gKRope.SetGlobalBuffer((__gm__ ElementK *)params.kRope);
        AscendC::GlobalTensor<int32_t> gblockTable;
        gblockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
#elif defined(__DAV_VEC__)
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
        AscendC::GlobalTensor<ElementOTmp> gOCoreTmp;
        gOCoreTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oCoreTmp);
        AscendC::GlobalTensor<ElementOTmp> gl;
        gl.SetGlobalBuffer((__gm__ ElementOTmp *)params.l);
        AscendC::GlobalTensor<float> gTilingFp64;
        gTilingFp64.SetGlobalBuffer((__gm__ float *)params.tiling);
#endif
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);

#if defined(__DAV_CUBE__)
        uint32_t coreIdx = AscendC::GetBlockIdx();
#elif defined(__DAV_VEC__)
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        float tor = gTilingFp64.GetValue(TILING_TOR);
#endif

        uint32_t coreNum = AscendC::GetBlockNum();
        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        uint32_t maxNumBlocksPerQuery = gTiling.GetValue(TILING_MAXBLOCKS);
        uint32_t totalTaskNumSpec = gTiling.GetValue(TILING_TOTAL_QTOKENS);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        uint32_t maxKvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);
        uint32_t formerTaskNum = gTiling.GetValue(TILING_FORMERTASKNUM);
        uint32_t tailTaskNum = gTiling.GetValue(TILING_TAILTASKNUM);
        uint32_t tailProcessNum = gTiling.GetValue(TILING_PROCESSNUM);

        uint32_t embed = NUM512;
        uint32_t embedRope = NUM64;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
        uint32_t kvHeads = NUM1;
        uint32_t strideQO = qHeads * embed;
        uint32_t strideQORope = qHeads * embedRope;
        uint32_t glFlag[2] = {1,1};
#if defined(__DAV_CUBE__)
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource);
#elif defined(__DAV_VEC__)
        EpilogueMLASoftmax epilogueMLATP1Softmax(resource, tor, maxKvSplitCoreNum);
        EpilogueMLARescaleO epilogueMLATP1RescaleO(resource, maxKvSplitCoreNum);
        uint32_t taskPingPongFlag = 0;
#endif
        // Go through tail task
        bool kerFlag = true;
        bool isFirstTask = true;
        uint32_t pingpongIdx = 0;
        uint32_t pvLoopPingpongIdx = 0;
        uint32_t qkInLoopPingpongIdx = 0;
        uint32_t qkLoopPingpongIdx = 0;
        uint32_t tailtaskIdx = 0;
        for (uint32_t process = coreIdx; process < tailProcessNum; process += uint32_t(coreNum)) {
            while (tailtaskIdx < tailTaskNum && process >= gTiling.GetValue(CUTASK_START_OFFSET + tailtaskIdx + 1)) {
                tailtaskIdx++;
            }
            // Get the offset of each core on the GM
            uint32_t taskIdx = tailtaskIdx + formerTaskNum;
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * taskIdx;
            uint32_t curBatch = gTiling.GetValue(offsetTiling);
            uint32_t curTokenWiseOffset = gTiling.GetValue(offsetTiling + 1);
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 2);
            uint32_t kvSplitPerCore = gTiling.GetValue(offsetTiling + 15);
            uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);
            uint64_t gmOffsetQ = (uint64_t)(curTokenWiseOffset * strideQO);
            uint64_t gmOffsetQRope = (uint64_t)(curTokenWiseOffset * strideQORope);
            uint64_t gmOffsetO = curTokenWiseOffset * qHeads * embed;

            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t kvSeqlenAlign = RoundUp(kvSeqlen, blockSize);
            uint32_t temp = gTiling.GetValue(CUTASK_START_OFFSET + tailtaskIdx);
            uint32_t curNIdx = process - temp;
            uint32_t curKVSeqlen = kvSplitPerCore;
            if (curNIdx >= kvSplitCoreNum) {
                continue;
            }
            if (curNIdx == (kvSplitCoreNum - 1)) {
                curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;
            }
            uint32_t startKV = curNIdx * kvSplitPerCore;
            uint32_t nLoop = (curKVSeqlen + blockSize - 1) / blockSize;
            uint32_t stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
            uint32_t rowNum = qHeads;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);
            uint64_t gmOffsetBlockTable = curBatch * maxNumBlocksPerQuery;
#if defined(__DAV_VEC__)
            uint32_t oFdOffset = 0;
            uint32_t lOffset = 0;
            if (maxKvSplitCoreNum != 1) {
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + 11);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + 12);
                uint64_t lAddr = (uint64_t)((((uint64_t)lAddrHigh32) << 32) | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + 13);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + 14);
                uint64_t fdAddr = (uint64_t)((((uint64_t)oFdAddrHigh32) << 32) | oFdAddrLow32);
                uint32_t headIdx = AscendC::GetSubBlockIdx() * qHeads / 2;
                oFdOffset = fdAddr * maxKvSplitCoreNum + headIdx * embed * maxKvSplitCoreNum + curNIdx * embed;
                lOffset = lAddr + headIdx * maxKvSplitCoreNum + curNIdx;
            }
#endif

            uint32_t nIdx = 0;
            if (isFirstTask) {
                if (nIdx < nLoop) {
                    if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop - 1) {
                        stackSeqTile = curKVSeqlen - nIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    // Calculate 4 blocks of Q * K^T
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                    uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, stackSeqTile);
                    LayoutK layoutKRope(embedRope, stackSeqTile);
                    LayoutS layoutS(rowNumRound, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                    uint64_t gmOffseS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;

                    blockMmadQK(gQ[gmOffsetQ], gQRope[gmOffsetQRope], gK, gKRope,
                                gblockTable[gmOffsetBlockTable + startKV / blockSize],
                                gS[gmOffseS], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK,
                                nIdx, nLoop, blockSize, curKVSeqlen, qkInLoopPingpongIdx, qkLoopPingpongIdx);

                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutS layoutS(rowNum, stackSeqTile, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    uint32_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                         (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         (uint64_t)gSPingPongFlag * TMP_SIZE;
                    uint32_t gmOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                         (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;

                    // Softmax one-stage calculation
                    epilogueMLATP1Softmax(gP[gmOffsetP], gS[gmOffsetS], layoutP,
                                          layoutS, actualBlockShapeQK, nIdx, glFlag, taskPingPongFlag, gSPingPongFlag);

                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    pingpongIdx += UNIT_BLOCK_STACK_NUM;
                    isFirstTask = false;
                }
            }

            // Split k seqlen
            for (nIdx = nIdx + UNIT_BLOCK_STACK_NUM; nIdx < nLoop + UNIT_BLOCK_STACK_NUM; nIdx += UNIT_BLOCK_STACK_NUM, pingpongIdx += UNIT_BLOCK_STACK_NUM) {
                if (nIdx < nLoop) {
                    if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop - 1) {
                        // Calculate the size of the tail block
                        stackSeqTile = curKVSeqlen - nIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                    uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                    // Calculate Q * K^T
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, stackSeqTile);
                    LayoutK layoutKRope(embedRope, stackSeqTile);
                    LayoutS layoutS(rowNumRound, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                    uint64_t gmOffseS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    blockMmadQK(gQ[gmOffsetQ], gQRope[gmOffsetQRope], gK, gKRope,
                                gblockTable[gmOffsetBlockTable + startKV / blockSize],
                                gS[gmOffseS], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK,
                                nIdx, nLoop, blockSize, curKVSeqlen,qkInLoopPingpongIdx, qkLoopPingpongIdx);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutS layoutS(rowNum, stackSeqTile, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    uint32_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                         (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         (uint64_t)gSPingPongFlag * TMP_SIZE;
                    uint32_t gmOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                         (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    // Softmax one-stage calculation
                    epilogueMLATP1Softmax(gP[gmOffsetP], gS[gmOffsetS], layoutP,
                                          layoutS, actualBlockShapeQK, nIdx, glFlag, taskPingPongFlag, gSPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }

                if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop + UNIT_BLOCK_STACK_NUM - 1) {
                    bool nextKerFlag = kerFlag;
                    uint32_t nextTailtaskIdx = tailtaskIdx;
                    for (uint32_t nextProcess = uint32_t(coreNum) + process; nextProcess < tailProcessNum; nextProcess += uint32_t(coreNum)) {
                        while (nextTailtaskIdx < tailTaskNum && nextProcess >= gTiling.GetValue(CUTASK_START_OFFSET + nextTailtaskIdx + 1)) {
                            nextTailtaskIdx++;
                        }

                        // Get the next offset of each core on the GM
                        uint32_t nextTaskIdx = nextTailtaskIdx + formerTaskNum;
                        uint32_t nextOffsetTiling = tilingHeadSize + tilingParaSize * nextTaskIdx;
                        uint32_t nextBatch = gTiling.GetValue(nextOffsetTiling);
                        uint32_t nextTokenWiseOffset = gTiling.GetValue(nextOffsetTiling + 1);
                        uint32_t nextKvSeqlen = gTiling.GetValue(nextOffsetTiling + 2);
                        uint32_t nextKvSplitPerCore = gTiling.GetValue(nextOffsetTiling + 15);
                        uint32_t nextKvSplitCoreNum = gTiling.GetValue(nextOffsetTiling + 16);
                        uint64_t nextGmOffsetQ = (uint64_t)(nextTokenWiseOffset * strideQO);
                        uint64_t nextGmOffsetQRope = (uint64_t)(nextTokenWiseOffset * strideQORope);
                        uint64_t nextGmOffsetO = nextTokenWiseOffset * qHeads * embed;

                        if (nextKvSeqlen == 0) {
                            continue;
                        }
                        uint32_t nextKvSeqlenAlign = RoundUp(nextKvSeqlen, blockSize);
                        uint32_t nextTaskStart = gTiling.GetValue(CUTASK_START_OFFSET + nextTailtaskIdx);
                        uint32_t nextCurNIdx = nextProcess - nextTaskStart;
                        uint32_t nextCurKVSeqlen = nextKvSplitPerCore;
                        if (nextCurNIdx == (nextKvSplitCoreNum - 1)) {
                            nextCurKVSeqlen = nextKvSeqlen - nextCurNIdx * nextKvSplitPerCore;
                        }
                        uint32_t nextNLoop = (nextCurKVSeqlen + blockSize - 1) / blockSize;
                        uint32_t nextStartKV = nextCurNIdx * nextKvSplitPerCore;
                        uint64_t nextGmOffsetBlockTable = nextBatch * maxNumBlocksPerQuery;
                        uint32_t nextNIdx = 0;
                        if (nextNIdx < nextNLoop) {
                            if (nextNIdx + UNIT_BLOCK_STACK_NUM > nextNLoop - 1) {
                                stackSeqTile = nextCurKVSeqlen - nextNIdx * blockSize;
                            } else {
                                stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                            }
                            // Calculate 4 blocks of Q * K^T
                            uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                            uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                            LayoutQ nextLayoutQ(rowNum, embed);
                            LayoutQ nextLayoutQRope(rowNum, embedRope);
                            LayoutK nextLayoutK(embed, stackSeqTile);
                            LayoutK nextLayoutKRope(embedRope, stackSeqTile);
                            LayoutS nextLayoutS(rowNumRound, stackSeqTileRound);
                            GemmCoord nextActualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                            uint64_t nextGmOffseS =
                                (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                            // Calculate Q * K^T
                            blockMmadQK(gQ[nextGmOffsetQ], gQRope[nextGmOffsetQRope], gK, gKRope,
                                        gblockTable[nextGmOffsetBlockTable + nextStartKV / blockSize],
                                        gS[nextGmOffseS], nextLayoutQ, nextLayoutQRope, nextLayoutK, nextLayoutKRope, nextLayoutS, nextActualBlockShapeQK,
                                        nextNIdx, nextNLoop, blockSize, nextCurKVSeqlen, qkInLoopPingpongIdx, qkLoopPingpongIdx);

                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                            LayoutP nextLayoutP(rowNum, stackSeqTile, stackSeqTileRound);
                            LayoutS nextLayoutS(rowNum, stackSeqTile, stackSeqTileRound);
                            GemmCoord nextActualBlockShapeQK{rowNum, stackSeqTile, embed};
                            uint32_t nextGmOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                                        (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                                        (uint64_t)gSPingPongFlag * TMP_SIZE;
                            uint32_t nextGmOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                                        (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                                        (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                            uint32_t nextTaskpingpongFlag = 1 - taskPingPongFlag;

                            uint64_t gmOffsetS =
                                (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;

                            // Softmax one-stage calculation
                            epilogueMLATP1Softmax(gP[nextGmOffsetP], gS[nextGmOffsetS], nextLayoutP,
                                                    nextLayoutS, nextActualBlockShapeQK, nextNIdx, glFlag, nextTaskpingpongFlag, gSPingPongFlag);
                            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                            break;
                        }
                    }
                }


                // Wait for the four Q * K^T calculations to complete before calculating P * V
                if (nIdx >= UNIT_BLOCK_STACK_NUM) {
                    if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop + UNIT_BLOCK_STACK_NUM - 1) {
                        // Calculate the size of the tail block
                        stackSeqTile = curKVSeqlen - (nIdx - UNIT_BLOCK_STACK_NUM) * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
#if defined(__DAV_CUBE__)
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutV layoutV(stackSeqTile, embed);
                    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t gPPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM - 1) % 2;
                    uint64_t gmOffseP = (uint64_t)coreIdx * TMP_SIZE * 2 + (uint64_t)gPPingPongFlag * TMP_SIZE;
                    uint64_t gmOffseOtmp = gmOffseP;
                    // Calculate P * V
                    blockMmadPV(gP[gmOffseP], gK, gblockTable[gmOffsetBlockTable + startKV / blockSize],
                                gOTmp[gmOffseOtmp], layoutP, layoutV,
                                layoutOTmp, actualBlockShapePV, nIdx, gPPingPongFlag, nLoop, blockSize, curKVSeqlen, softmaxReady, pvLoopPingpongIdx);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#elif defined(__DAV_VEC__)
                    // Wait for P * V calculation to complete
                    uint32_t rescaleOPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM - 1) % 2;
                    Arch::CrossCoreWaitFlag(pvReady);
                    LayoutO layoutO(rowNum, embed);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t isLastNTile = (nIdx >= nLoop) ? 1 : 0;
                    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
                    // Softmax two-stage update
                    epilogueMLATP1RescaleO(gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO],
                                           gOCoreTmp[oFdOffset], gl[lOffset], layoutOTmp,
                                           layoutUpdate, layoutO, actualBlockShapePV, nIdx, isLastNTile,
                                           rescaleOPingPongFlag, glFlag, taskPingPongFlag);
#endif
                }
            }
#if defined(__DAV_VEC__)
            taskPingPongFlag = 1 - taskPingPongFlag;
#endif
        }

        icache_preload(1);
#if defined(__DAV_VEC__)
        epilogueMLATP1Softmax.SetkvSplitCoreNum(1);
        epilogueMLATP1RescaleO.SetkvSplitCoreNum(1);
#endif
        // Go through former task
        bool isForward = true;
        pingpongIdx = 0;
        for (uint32_t process = coreIdx; process < formerTaskNum; process += uint32_t(coreNum)) {
            uint32_t bigProcess = process - (process % coreNum) + (coreNum - 1);
            bigProcess = (bigProcess > formerTaskNum - 1) ? (formerTaskNum - 1) : bigProcess;
            uint32_t realProcess = isForward ? process : (bigProcess - process % uint32_t(coreNum));
            isForward = !isForward;

            // Get the offset of each core on the GM
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * realProcess;
            uint32_t curBatch = gTiling.GetValue(offsetTiling);
            uint32_t curTokenWiseOffset = gTiling.GetValue(offsetTiling + 1);
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 2);
            uint64_t gmOffsetQ = (uint64_t)(curTokenWiseOffset * strideQO);
            uint64_t gmOffsetQRope = (uint64_t)(curTokenWiseOffset * strideQORope);
            uint64_t gmOffsetO = curTokenWiseOffset * qHeads * embed;

            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t nLoop = (kvSeqlen + blockSize - 1) / blockSize;
            uint32_t stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
            uint32_t rowNum = qHeads;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);
            uint64_t gmOffsetBlockTable = curBatch * maxNumBlocksPerQuery;

            if (process == coreIdx) {
                uint32_t nIdx = 0;
                if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop - 1) {
                    stackSeqTile = kvSeqlen - nIdx * blockSize;
                } else {
                    stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                }
                // Calculate 4 blocks of Q * K^T
                uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                LayoutQ layoutQ(rowNum, embed);
                LayoutQ layoutQRope(rowNum, embedRope);
                LayoutK layoutK(embed, stackSeqTile);
                LayoutK layoutKRope(embedRope, stackSeqTile);
                LayoutS layoutS(rowNumRound, stackSeqTileRound);
                GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                uint64_t gmOffseS =
                    (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                // Calculate Q * K^T
                blockMmadQK(gQ[gmOffsetQ], gQRope[gmOffsetQRope], gK, gKRope, gblockTable[gmOffsetBlockTable],
                            gS[gmOffseS], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK,
                            nIdx, nLoop, blockSize, kvSeqlen, qkInLoopPingpongIdx, qkLoopPingpongIdx);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                LayoutS layoutS(rowNum, stackSeqTile, stackSeqTileRound);
                GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                uint32_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                        (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                        (uint64_t)((pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2) * TMP_SIZE;
                uint32_t gmOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                        (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                        (uint64_t)((pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2) * TMP_SIZE_DECODER * 2;
                // Softmax one-stage calculation
                epilogueMLATP1Softmax(gP[gmOffsetP], gS[gmOffsetS], layoutP,
                                        layoutS, actualBlockShapeQK, nIdx, glFlag, taskPingPongFlag, gSPingPongFlag);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                pingpongIdx += UNIT_BLOCK_STACK_NUM;
            }

            // Split k seqlen
            for (uint32_t nIdx = UNIT_BLOCK_STACK_NUM; nIdx < nLoop + UNIT_BLOCK_STACK_NUM; nIdx += UNIT_BLOCK_STACK_NUM, pingpongIdx += UNIT_BLOCK_STACK_NUM) {
                if (nIdx < nLoop) {
                    if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop - 1) {
                        // Calculate the size of the tail block
                        stackSeqTile = kvSeqlen - nIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                    uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                    // Calculate Q * K^T
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, stackSeqTile);
                    LayoutK layoutKRope(embedRope, stackSeqTile);
                    LayoutS layoutS(rowNumRound, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                    uint64_t gmOffseS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    blockMmadQK(gQ[gmOffsetQ], gQRope[gmOffsetQRope], gK, gKRope, gblockTable[gmOffsetBlockTable],
                                gS[gmOffseS], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK,
                                nIdx, nLoop, blockSize, kvSeqlen, qkInLoopPingpongIdx, qkLoopPingpongIdx);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutS layoutS(rowNum, stackSeqTile, stackSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    uint32_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                         (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         gSPingPongFlag * TMP_SIZE;
                    uint32_t gmOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                         (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                         gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    // Softmax one-stage calculation
                    epilogueMLATP1Softmax(gP[gmOffsetP], gS[gmOffsetS], layoutP,
                                          layoutS, actualBlockShapeQK, nIdx, glFlag, taskPingPongFlag, gSPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                uint32_t nextProcess = process + uint32_t(coreNum);
                if(nIdx + UNIT_BLOCK_STACK_NUM > nLoop + UNIT_BLOCK_STACK_NUM - 1 && nextProcess < formerTaskNum) {
                    bool nextIsForward = isForward;
                    uint32_t nextBigProcess = nextProcess - (nextProcess % coreNum) + (coreNum - 1);
                    nextBigProcess = (nextBigProcess > formerTaskNum - 1) ? (formerTaskNum - 1) : nextBigProcess;
                    uint32_t nextRealProcess = nextIsForward ? nextProcess : (nextBigProcess - nextProcess % uint32_t(coreNum));

                    // Get the offset of each core on the GM
                    uint32_t nextOffsetTiling = tilingHeadSize + tilingParaSize * nextRealProcess;
                    uint32_t nextCurBatch = gTiling.GetValue(nextOffsetTiling);
                    uint32_t nextCurTokenWiseOffset = gTiling.GetValue(nextOffsetTiling + 1);
                    uint32_t nextKvSeqlen = gTiling.GetValue(nextOffsetTiling + 2);
                    uint32_t nextNLoop = (nextKvSeqlen + blockSize - 1) / blockSize;
                    uint64_t nextGMOffsetQ = (uint64_t)(nextCurTokenWiseOffset * strideQO);
                    uint64_t nextGMOffsetQRope = (uint64_t)(nextCurTokenWiseOffset * strideQORope);
                    uint64_t nextGMOffsetBlockTable = nextCurBatch * maxNumBlocksPerQuery;
                    uint32_t nextNIdx = 0;
                    if (nextNIdx + UNIT_BLOCK_STACK_NUM > nextNLoop - 1) {
                        // Calculate the size of the tail block
                        stackSeqTile = nextKvSeqlen - nextNIdx * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    // Calculate 4 blocks of Q * K^T
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
                    uint32_t gSPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM) % 2;
#if defined(__DAV_CUBE__)
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, stackSeqTile);
                    LayoutK layoutKRope(embedRope, stackSeqTile);
                    LayoutS layoutS(rowNumRound, stackSeqTileRound);
                    GemmCoord nextActualBlockShapeQK{rowNum, stackSeqTile, embed + embedRope};
                    uint64_t nextGMOffseS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER * 4 + (uint64_t)gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    // Calculate Q * K^T
                    blockMmadQK(gQ[nextGMOffsetQ], gQRope[nextGMOffsetQRope], gK, gKRope, gblockTable[nextGMOffsetBlockTable],
                                gS[nextGMOffseS], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, nextActualBlockShapeQK,
                                nextNIdx, nextNLoop, blockSize, nextKvSeqlen, qkInLoopPingpongIdx, qkLoopPingpongIdx);

                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#elif defined(__DAV_VEC__)
                    LayoutP nextLayoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutS nextLayoutS(rowNum, stackSeqTile, stackSeqTileRound);
                    GemmCoord nextActualBlockShapeQK{rowNum, stackSeqTile, embed};
                    uint32_t nextTaskPingPongFlag = 1 - taskPingPongFlag;
                    uint32_t nextGMOffsetP = (uint64_t)coreIdx * TMP_SIZE * 2 +
                                        (uint64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                        gSPingPongFlag * TMP_SIZE;
                    uint32_t nextGMOffsetS = (int64_t)coreIdx * TMP_SIZE_DECODER * 4 +
                                        (int64_t)subBlockIdx * rowNum / 2 * stackSeqTileRound +
                                        gSPingPongFlag * TMP_SIZE_DECODER * 2;
                    // Softmax one-stage calculation
                    epilogueMLATP1Softmax(gP[nextGMOffsetP], gS[nextGMOffsetS], nextLayoutP,
                                        nextLayoutS, nextActualBlockShapeQK, nextNIdx, glFlag, nextTaskPingPongFlag, gSPingPongFlag);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }
                // Wait for the four Q * K^T calculations to complete before calculating P * V
                if (nIdx >= UNIT_BLOCK_STACK_NUM) {
                    if (nIdx + UNIT_BLOCK_STACK_NUM > nLoop + UNIT_BLOCK_STACK_NUM - 1) {
                        // Calculate the size of the tail block
                        stackSeqTile = kvSeqlen - (nIdx - UNIT_BLOCK_STACK_NUM) * blockSize;
                    } else {
                        stackSeqTile = blockSize * UNIT_BLOCK_STACK_NUM;
                    }
                    uint32_t stackSeqTileRound = RoundUp<BLOCK_SIZE>(stackSeqTile);
#if defined(__DAV_CUBE__)
                    LayoutP layoutP(rowNum, stackSeqTile, stackSeqTileRound);
                    LayoutV layoutV(stackSeqTile, embed);
                    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t gPPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM - 1) % 2;
                    uint64_t gmOffseP = (uint64_t)coreIdx * TMP_SIZE * 2 + (uint64_t)gPPingPongFlag * TMP_SIZE;
                    uint64_t gmOffseOtmp = gmOffseP;
                    // Calculate P * V
                    blockMmadPV(gP[gmOffseP], gK, gblockTable[gmOffsetBlockTable], gOTmp[gmOffseOtmp], layoutP, layoutV,
                                layoutOTmp, actualBlockShapePV, nIdx, gPPingPongFlag, nLoop, blockSize, kvSeqlen, softmaxReady, pvLoopPingpongIdx);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#elif defined(__DAV_VEC__)
                    // Wait for P * V calculation to complete
                    uint32_t rescaleOPingPongFlag = (pingpongIdx / UNIT_BLOCK_STACK_NUM - 1) % 2;
                    Arch::CrossCoreWaitFlag(pvReady);
                    LayoutO layoutO(rowNum, embed);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    uint32_t isLastNTile = (nIdx >= nLoop) ? 1 : 0;
                    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
                    // Softmax two-stage update
                    epilogueMLATP1RescaleO(gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO],
                                           gOCoreTmp[0], gl[0], layoutOTmp,
                                           layoutUpdate, layoutO, actualBlockShapePV, nIdx, isLastNTile,
                                           rescaleOPingPongFlag, glFlag, taskPingPongFlag);
#endif
                }
            }
#if defined(__DAV_VEC__)
            taskPingPongFlag = 1 - taskPingPongFlag;
#endif
        }

#if defined(__DAV_CUBE__)
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#elif defined(__DAV_VEC__)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        // flash decoding
        if (maxKvSplitCoreNum != 1) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

            AscendC::SetAtomicNone();
            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

            EpilogueMLAFDRescaleO epilogueMLAFDRescaleO(resource, maxKvSplitCoreNum);

            uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
            uint32_t aivId = AscendC::GetBlockIdx();

            uint32_t headsProcess =
                (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX
                    ? HEADS_PROCESS_MAX
                    : (COMPUTE_ELE_NUM / embed);
            uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
            uint32_t loopsTotal = tailTaskNum * loopsPerBatch;

            for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
                uint32_t taskIdx = loopIdx / loopsPerBatch + formerTaskNum;
                uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;

                uint32_t offsetTiling = tilingHeadSize + tilingParaSize * taskIdx;
                uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 2);
                uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);

                if (kvSeqlen == 0) {
                    continue;
                }

                uint32_t curTokenWiseOffset = gTiling.GetValue(offsetTiling + 1);
                uint64_t oAddr = curTokenWiseOffset * qHeads * embed;
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + 11);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + 12);
                uint64_t lOffset = (uint64_t)((((uint64_t)lAddrHigh32) << 32) | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + 13);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + 14);
                uint64_t oFdOffset = (uint64_t)((((uint64_t)oFdAddrHigh32) << 32) | oFdAddrLow32);

                uint32_t actualHeads = headsProcess;
                if (loopIdxInBatch == loopsPerBatch - 1) {
                    actualHeads = qHeads - loopIdxInBatch * headsProcess;
                }

                epilogueMLAFDRescaleO(
                    gO[oAddr + loopIdxInBatch * headsProcess * embed],
                    gOCoreTmp[oFdOffset * maxKvSplitCoreNum +
                              loopIdxInBatch * headsProcess * maxKvSplitCoreNum * embed],
                    gl[lOffset + loopIdxInBatch * headsProcess * maxKvSplitCoreNum],
                    actualHeads, headsProcess, embed, kvSplitCoreNum);
            }
        }
#endif
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

template <class Dtype>
CATLASS_GLOBAL void MLATp1Spec(uint64_t hardwareSyncAddr,
                                GM_ADDR q,
                                GM_ADDR qRope,
                                GM_ADDR k,
                                GM_ADDR kRope,
                                GM_ADDR blockTables,
                                GM_ADDR o,
                                GM_ADDR s,
                                GM_ADDR p,
                                GM_ADDR oTmp,
                                GM_ADDR oUpdate,
                                GM_ADDR oCoreTmp,
                                GM_ADDR l,
                                GM_ADDR tiling)
{
    // Set hardware sync address
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);

    using ArchTag = Arch::AtlasA2;
    using ElementQ = Dtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = Dtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = Dtype;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = Dtype;
    using LayoutP = layout::RowMajor;
    using ElementO = Dtype;
    using LayoutO = layout::RowMajor;
    using ElementMask = Dtype;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 576>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQKTp1Spec;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLATP1Softmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPVTp1Spec;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLATP1RescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel = MLAKernelTp1Spec<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax,
                                       EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}
