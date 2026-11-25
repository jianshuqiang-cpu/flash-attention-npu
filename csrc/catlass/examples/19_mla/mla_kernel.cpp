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
template <
    class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax, class EpilogueMLARescaleO,
    class EpilogueMLAFDRescaleO>
class MLAKernel {
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
        Params()
        {}

        CATLASS_DEVICE
        Params(
            GM_ADDR q_, GM_ADDR qRope_, GM_ADDR k_, GM_ADDR kRope_, GM_ADDR blockTables_, GM_ADDR o_, GM_ADDR s_,
            GM_ADDR p_, GM_ADDR oTmp_, GM_ADDR oUpdate_, GM_ADDR oCoreTmp_, GM_ADDR l_, GM_ADDR tiling_)
            : q(q_),
              qRope(qRope_),
              k(k_),
              kRope(kRope_),
              blockTables(blockTables_),
              o(o_),
              s(s_),
              p(p_),
              oTmp(oTmp_),
              oUpdate(oUpdate_),
              oCoreTmp(oCoreTmp_),
              l(l_),
              tiling(tiling_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    MLAKernel()
    {}

    CATLASS_DEVICE void operator()(Params const& params)
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
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
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t*)params.tiling);

#ifdef __DAV_CUBE__
        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementQ> gQRope;
        gQRope.SetGlobalBuffer((__gm__ ElementQ*)params.qRope);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gKRope;
        gKRope.SetGlobalBuffer((__gm__ ElementK*)params.kRope);
        AscendC::GlobalTensor<int32_t> gblockTable;
        gblockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTables));

        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource);
#endif
#ifdef __DAV_VEC__
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)params.oUpdate);
        AscendC::GlobalTensor<ElementOTmp> gOCoreTmp;
        gOCoreTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oCoreTmp);
        AscendC::GlobalTensor<ElementOTmp> gl;
        gl.SetGlobalBuffer((__gm__ ElementOTmp*)params.l);
        AscendC::GlobalTensor<float> gTilingFp64;
        gTilingFp64.SetGlobalBuffer((__gm__ float*)params.tiling);
#endif

        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t embed = gTiling.GetValue(TILING_HEADDIM);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        uint32_t curQheadSplitSize = gTiling.GetValue(TILING_HEAD_SPLIT_SIZE);
        uint32_t curQheadSplitNum = gTiling.GetValue(TILING_HEAD_SPLIT_NUM);
        uint32_t maxKvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);

        uint32_t strideQO = qHeads * embed;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
#ifdef __DAV_VEC__
        float tor = gTilingFp64.GetValue(TILING_TOR);
        uint32_t glFlag[2] = {1, 1};

        EpilogueMLASoftmax epilogueMLASoftmax(resource, tor, maxKvSplitCoreNum);
        EpilogueMLARescaleO epilogueMLARescaleO(resource, maxKvSplitCoreNum);
#endif

#ifdef __DAV_CUBE__
        uint32_t embedRope = gTiling.GetValue(TILING_HEADDIM_ROPE);
        uint32_t maxNumBlocksPerQuery = gTiling.GetValue(TILING_MAXBLOCKS);
        uint32_t kvHeads = gTiling.GetValue(TILING_KVHEADS);
        uint32_t strideQORope = qHeads * embedRope;
        uint32_t strideKV = static_cast<uint64_t>(kvHeads) * embed;
        uint32_t strideKVRope = static_cast<uint64_t>(kvHeads) * embedRope;
#endif

        uint32_t coreIdx = 0;
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        uint32_t coreNum = AscendC::GetBlockNum();
        // When the tiling decides to split kv it publishes a non-zero TILING_PROCESSNUM together
        // with a cumulative-task prefix-sum table (CUTASK_START_OFFSET ...). In that case the
        // process -> task mapping is the dense T+1-style mapping; otherwise (no kv split) fall
        // back to the original back-and-forth (kerFlag / isForward) dispatch.
        uint32_t tilingProcessNum = gTiling.GetValue(TILING_PROCESSNUM);
        uint32_t processNum = (tilingProcessNum != 0) ? tilingProcessNum : batch * curQheadSplitNum * maxKvSplitCoreNum;

        bool kerFlag = true;
        bool isForward = true;
        uint32_t pingpongIdx = 0;
        bool isFirstTask = true;
        uint32_t taskIdx = 0;
#ifdef __DAV_CUBE__
        uint32_t locPingPongIdx = 0;
#endif
#ifdef __DAV_VEC__
        uint32_t taskPingPongFlag = 0;
#endif

        // Cube and vector share the same task traversal; stage bodies stay in macro branches.
        for (uint32_t process = coreIdx; process < processNum; process += uint32_t(coreNum)) {
            uint32_t curBatch;
            if (tilingProcessNum != 0) {
                // T+1-style continuous (prefix-sum) task mapping
                while (taskIdx < batch && process >= gTiling.GetValue(CUTASK_START_OFFSET + taskIdx + 1)) {
                    taskIdx++;
                }
                curBatch = taskIdx;
            } else {
                uint32_t bigProcess = process - (process % coreNum) + (coreNum - 1);
                bigProcess = (bigProcess > processNum - 1) ? (processNum - 1) : bigProcess;
                uint32_t realProcess = isForward ? process : (bigProcess - process % coreNum);
                isForward = !isForward;
                curBatch = realProcess / (curQheadSplitNum * maxKvSplitCoreNum);
            }
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
            uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
            uint32_t kvSplitPerCore = gTiling.GetValue(offsetTiling + 15);
            uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);

            if (kvSeqlen == 0) {
                continue;
            }

            uint32_t qHeadSplitIdx;
            uint32_t curNIdx;
            if (tilingProcessNum != 0) {
                qHeadSplitIdx = 0;
                curNIdx = process - gTiling.GetValue(CUTASK_START_OFFSET + curBatch);
            } else {
                qHeadSplitIdx = (process % (curQheadSplitNum * maxKvSplitCoreNum)) / maxKvSplitCoreNum;
                curNIdx = process % maxKvSplitCoreNum;
                if (kerFlag) {
                    kerFlag = false;
                } else {
                    kerFlag = true;
                    curNIdx = maxKvSplitCoreNum - curNIdx - 1;
                }
            }
            uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1)) ?
                                                (qHeads - qHeadSplitIdx * curQheadSplitSize) :
                                                curQheadSplitSize;
            uint32_t curStartHeadIdx = qHeadSplitIdx * curQheadSplitSize;
            uint32_t curKVSeqlen = kvSplitPerCore;

            if (curNIdx >= kvSplitCoreNum) {
                continue;
            }
            if (curNIdx == (kvSplitCoreNum - 1)) {
                curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;
            }

            uint32_t tokenNumPerHead = qSeqlen;
            uint32_t seqTile = blockSize;
            uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;
            uint32_t kSeqTile = seqTile;
            uint32_t kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            uint32_t vSeqTile = seqTile;
            uint32_t vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);

            uint32_t rowNum = qHeadSplitSizeActual * tokenNumPerHead;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

#ifdef __DAV_CUBE__
            uint32_t realCurBatch = gTiling.GetValue(offsetTiling + 2);
            uint32_t qAddrHigh = gTiling.GetValue(offsetTiling + 4);
            uint32_t qAddrLow = gTiling.GetValue(offsetTiling + 5);
            uint64_t qAddr = (uint64_t)((((uint64_t)qAddrHigh) << 32) | qAddrLow);
            uint32_t qRopeAddrHigh = gTiling.GetValue(offsetTiling + 6);
            uint32_t qRopeAddrLow = gTiling.GetValue(offsetTiling + 7);
            uint64_t qRopeAddr = (uint64_t)((((uint64_t)qRopeAddrHigh) << 32) | qRopeAddrLow);
            uint64_t gQOffset = qAddr + curStartHeadIdx * embed;
            uint64_t gQRopeOffset = qRopeAddr + curStartHeadIdx * embedRope;
            uint32_t startKV = curNIdx * kvSplitPerCore;
#endif
#ifdef __DAV_VEC__
            uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + 4);
            uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + 5);
            uint64_t oAddr = (uint64_t)((((uint64_t)oAddrHigh32) << 32) | oAddrLow32);
            uint64_t gmOffsetO = oAddr + curStartHeadIdx * embed;
            uint32_t oFdOffset = 0;
            uint32_t lOffset = 0;
            if (maxKvSplitCoreNum != 1) {
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + 11);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + 12);
                uint64_t lAddr = (uint64_t)((((uint64_t)lAddrHigh32) << 32) | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + 13);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + 14);
                uint64_t FdAddr = (uint64_t)((((uint64_t)oFdAddrHigh32) << 32) | oFdAddrLow32);
                uint32_t headIdx = curStartHeadIdx + AscendC::GetSubBlockIdx() * qHeadSplitSizeActual / 2;
                oFdOffset = FdAddr * maxKvSplitCoreNum + headIdx * embed * maxKvSplitCoreNum + curNIdx * embed;
                lOffset = lAddr + headIdx * maxKvSplitCoreNum + curNIdx;
            }
#endif

            if (isFirstTask && nLoop > 0) {
                uint32_t nIdx = 0;
                if (nIdx == (nLoop - 1)) {
                    kSeqTile = (curKVSeqlen - nIdx * seqTile);
                    kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                }
#ifdef __DAV_CUBE__
                LayoutQ layoutQ(rowNum, embed);
                LayoutQ layoutQRope(rowNum, embedRope);
                LayoutK layoutK(embed, kSeqTile);
                LayoutK layoutKRope(embedRope, kSeqTile);
                LayoutS layoutS(rowNumRound, kSeqTileRound);
                GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
                MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
                uint32_t qkPingPongFlag = pingpongIdx % 2;
                int32_t blockTableId =
                    gblockTable.GetValue(realCurBatch * maxNumBlocksPerQuery + startKV / blockSize + nIdx);
                uint64_t kvOffset = (uint64_t)blockTableId * blockSize * strideKV;
                uint64_t kvOffsetRope = (uint64_t)blockTableId * blockSize * strideKVRope;
                uint64_t gSOffset =
                    (uint64_t)coreIdx * TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag * TMP_SIZE_DECODER / 2;
                blockMmadQK(
                    gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope], gS[gSOffset], layoutQ,
                    layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK, qShapeSingleNd, qHeads, nIdx,
                    pingpongIdx);

                pingpongIdx++;
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                isFirstTask = false;
#endif
#ifdef __DAV_VEC__
                uint32_t softmaxPingPongFlag = pingpongIdx % 2;
                Arch::CrossCoreWaitFlag(qkReady);

                LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound);
                LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
                GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
                uint64_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
                uint64_t gmOffsetS = (uint64_t)coreIdx * TMP_SIZE_DECODER + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
                epilogueMLASoftmax(
                    gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK, nIdx, qHeadSplitSizeActual,
                    softmaxPingPongFlag, glFlag, taskPingPongFlag);

                pingpongIdx++;
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                isFirstTask = false;
#endif
            }

            for (uint32_t nIdx = 1; nIdx < nLoop + 1; nIdx++, pingpongIdx++) {
                if (nIdx != nLoop) {
                    if (nIdx == (nLoop - 1)) {
                        kSeqTile = (curKVSeqlen - nIdx * seqTile);
                        kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                    }
#ifdef __DAV_CUBE__
                    LayoutQ layoutQ(rowNum, embed);
                    LayoutQ layoutQRope(rowNum, embedRope);
                    LayoutK layoutK(embed, kSeqTile);
                    LayoutK layoutKRope(embedRope, kSeqTile);
                    LayoutS layoutS(rowNumRound, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
                    MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
                    uint32_t qkPingPongFlag = pingpongIdx % 2;
                    int32_t blockTableId =
                        gblockTable.GetValue(realCurBatch * maxNumBlocksPerQuery + startKV / blockSize + nIdx);
                    uint64_t kvOffset = (uint64_t)blockTableId * blockSize * strideKV;
                    uint64_t kvOffsetRope = (uint64_t)blockTableId * blockSize * strideKVRope;
                    uint64_t gSOffset =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag * TMP_SIZE_DECODER / 2;
                    blockMmadQK(
                        gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope], gS[gSOffset], layoutQ,
                        layoutQRope, layoutK, layoutKRope, layoutS, actualBlockShapeQK, qShapeSingleNd, qHeads, nIdx,
                        pingpongIdx);

                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreWaitFlag(qkReady);

                    LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound);
                    LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
                    uint32_t softmaxPingPongFlag = pingpongIdx % 2;
                    uint64_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
                    uint64_t gmOffsetS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
                    epilogueMLASoftmax(
                        gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK, nIdx, qHeadSplitSizeActual,
                        softmaxPingPongFlag, glFlag, taskPingPongFlag);

                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                }

                if (nIdx == nLoop) {
                    bool nextKerFlag = kerFlag;
                    bool nextIsForward = isForward;
                    uint32_t nextTaskIdx = taskIdx;
                    for (uint32_t nextProcess = uint32_t(coreNum) + process; nextProcess < processNum;
                         nextProcess += uint32_t(coreNum)) {
                        uint32_t nextBatch;
                        if (tilingProcessNum != 0) {
                            while (nextTaskIdx < batch &&
                                   nextProcess >= gTiling.GetValue(CUTASK_START_OFFSET + nextTaskIdx + 1)) {
                                nextTaskIdx++;
                            }
                            nextBatch = nextTaskIdx;
                        } else {
                            uint32_t nextBigProcess = nextProcess - (nextProcess % coreNum) + (coreNum - 1);
                            nextBigProcess = (nextBigProcess > processNum - 1) ? (processNum - 1) : nextBigProcess;
                            uint32_t nextRealProcess =
                                nextIsForward ? nextProcess : (nextBigProcess - nextProcess % coreNum);
                            nextIsForward = !nextIsForward;
                            nextBatch = nextRealProcess / (curQheadSplitNum * maxKvSplitCoreNum);
                        }
                        uint32_t nextOffsetTiling = tilingHeadSize + tilingParaSize * nextBatch;
                        uint32_t nextQSeqlen = gTiling.GetValue(nextOffsetTiling);
                        uint32_t nextKVSeqlen = gTiling.GetValue(nextOffsetTiling + 1);
                        uint32_t nextKVSplitPerCore = gTiling.GetValue(nextOffsetTiling + 15);
                        uint32_t nextKVSplitCoreNum = gTiling.GetValue(nextOffsetTiling + 16);

                        if (nextKVSeqlen == 0) {
                            continue;
                        }

                        uint32_t nextQHeadSplitIdx;
                        uint32_t nextCurNIdx;
                        if (tilingProcessNum != 0) {
                            nextQHeadSplitIdx = 0;
                            nextCurNIdx = nextProcess - gTiling.GetValue(CUTASK_START_OFFSET + nextBatch);
                        } else {
                            nextQHeadSplitIdx =
                                (nextProcess % (curQheadSplitNum * maxKvSplitCoreNum)) / maxKvSplitCoreNum;
                            nextCurNIdx = nextProcess % maxKvSplitCoreNum;
                            if (nextKerFlag) {
                                nextKerFlag = false;
                            } else {
                                nextKerFlag = true;
                                nextCurNIdx = maxKvSplitCoreNum - nextCurNIdx - 1;
                            }
                        }
                        uint32_t nextQHeadSplitSizeActual = (nextQHeadSplitIdx == (curQheadSplitNum - 1)) ?
                                                                (qHeads - nextQHeadSplitIdx * curQheadSplitSize) :
                                                                curQheadSplitSize;
                        uint32_t nextCurKVSeqlen = nextKVSplitPerCore;

                        if (nextCurNIdx >= nextKVSplitCoreNum) {
                            continue;
                        }
                        if (nextCurNIdx == (nextKVSplitCoreNum - 1)) {
                            nextCurKVSeqlen = nextKVSeqlen - nextCurNIdx * nextKVSplitPerCore;
                        }

                        uint32_t nextTokenNumPerHead = nextQSeqlen;
                        uint32_t nextSeqTile = blockSize;
                        uint32_t nextNLoop = (nextCurKVSeqlen + nextSeqTile - 1) / nextSeqTile;
                        uint32_t nextKSeqTile = nextSeqTile;
                        uint32_t nextKSeqTileRound = RoundUp<BLOCK_SIZE>(nextKSeqTile);
                        uint32_t nextRowNum = nextQHeadSplitSizeActual * nextTokenNumPerHead;
                        uint32_t nextRowNumRound = RoundUp<BLOCK_SIZE>(nextRowNum);
                        uint32_t nextNIdx = 0;
                        if (nextNIdx == (nextNLoop - 1)) {
                            nextKSeqTile = (nextCurKVSeqlen - nextNIdx * nextSeqTile);
                            nextKSeqTileRound = RoundUp<BLOCK_SIZE>(nextKSeqTile);
                        }

#ifdef __DAV_CUBE__
                        uint32_t nextRealBatch = gTiling.GetValue(nextOffsetTiling + 2);
                        uint32_t nextQAddrHigh = gTiling.GetValue(nextOffsetTiling + 4);
                        uint32_t nextQAddrLow = gTiling.GetValue(nextOffsetTiling + 5);
                        uint64_t nextQAddr = (uint64_t)((((uint64_t)nextQAddrHigh) << 32) | nextQAddrLow);
                        uint32_t nextQRopeAddrHigh = gTiling.GetValue(nextOffsetTiling + 6);
                        uint32_t nextQRopeAddrLow = gTiling.GetValue(nextOffsetTiling + 7);
                        uint64_t nextQRopeAddr = (uint64_t)((((uint64_t)nextQRopeAddrHigh) << 32) | nextQRopeAddrLow);
                        uint32_t nextCurStartHeadIdx = nextQHeadSplitIdx * curQheadSplitSize;
                        uint64_t nextGQOffset = nextQAddr + nextCurStartHeadIdx * embed;
                        uint64_t nextGQRopeOffset = nextQRopeAddr + nextCurStartHeadIdx * embedRope;
                        uint32_t nextStartKV = nextCurNIdx * nextKVSplitPerCore;
                        LayoutQ nextLayoutQ(nextRowNum, embed);
                        LayoutQ nextLayoutQRope(nextRowNum, embedRope);
                        LayoutK nextLayoutK(embed, nextKSeqTile);
                        LayoutK nextLayoutKRope(embedRope, nextKSeqTile);
                        LayoutS nextLayoutS(nextRowNumRound, nextKSeqTileRound);
                        GemmCoord nextActualBlockShapeQK{nextRowNum, nextKSeqTile, embed + embedRope};
                        MatrixCoord nextQShapeSingleNd{nextQHeadSplitSizeActual, embed};
                        uint32_t qkPingPongFlag = pingpongIdx % 2;
                        int32_t nextBlockTableId = gblockTable.GetValue(
                            nextRealBatch * maxNumBlocksPerQuery + nextStartKV / blockSize + nextNIdx);
                        uint64_t nextKvOffset = (uint64_t)nextBlockTableId * blockSize * strideKV;
                        uint64_t nextKvOffsetRope = (uint64_t)nextBlockTableId * blockSize * strideKVRope;
                        uint64_t nextGSOffset =
                            (uint64_t)coreIdx * TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag * TMP_SIZE_DECODER / 2;
                        blockMmadQK(
                            gQ[nextGQOffset], gQRope[nextGQRopeOffset], gK[nextKvOffset], gKRope[nextKvOffsetRope],
                            gS[nextGSOffset], nextLayoutQ, nextLayoutQRope, nextLayoutK, nextLayoutKRope, nextLayoutS,
                            nextActualBlockShapeQK, nextQShapeSingleNd, qHeads, nextNIdx, pingpongIdx);

                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_VEC__
                        uint32_t softmaxPingPongFlag = pingpongIdx % 2;
                        uint32_t nextTaskPingPongFlag = 1 - taskPingPongFlag;
                        Arch::CrossCoreWaitFlag(qkReady);

                        LayoutP nextLayoutP(nextRowNum, nextKSeqTile, nextKSeqTileRound);
                        LayoutS nextLayoutS(nextRowNumRound, nextKSeqTile, nextKSeqTileRound);
                        GemmCoord nextActualBlockShapeQK{nextRowNum, nextKSeqTile, embedRound};
                        uint64_t nextGMOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
                        uint64_t nextGMOffsetS =
                            (uint64_t)coreIdx * TMP_SIZE_DECODER + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
                        epilogueMLASoftmax(
                            gP[nextGMOffsetP], gS[nextGMOffsetS], nextLayoutP, nextLayoutS, nextActualBlockShapeQK,
                            nextNIdx, nextQHeadSplitSizeActual, softmaxPingPongFlag, glFlag, nextTaskPingPongFlag);

                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                        break;
                    }
                }

                if (nIdx == nLoop) {
                    vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
                    vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
                }

#ifdef __DAV_CUBE__
                LayoutP layoutP(rowNum, vSeqTile, vSeqTileRound);
                LayoutV layoutV(embed, vSeqTile);
                LayoutOTmp layoutOTmp(rowNumRound, embedRound);
                GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                uint32_t pvPingPongFlag = (pingpongIdx - 1) % 2;
                uint64_t gPOffset = (uint64_t)coreIdx * TMP_SIZE + (uint64_t)pvPingPongFlag * TMP_SIZE / 2;
                uint64_t gOTmpOffset = (uint64_t)coreIdx * TMP_SIZE * 2 + (uint64_t)pvPingPongFlag * TMP_SIZE;
                blockMmadPV(
                    gP[gPOffset], gOTmp[gOTmpOffset], layoutP, layoutV, layoutOTmp, actualBlockShapePV, nIdx,
                    pingpongIdx, softmaxReady, locPingPongIdx);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_VEC__
                Arch::CrossCoreWaitFlag(pvReady);

                LayoutO layoutO(tokenNumPerHead, strideQO);
                LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                uint32_t rescaleOPingPongFlag = (pingpongIdx - 1) % 2;
                uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
                uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
                uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
                epilogueMLARescaleO(
                    gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO], gOCoreTmp[oFdOffset], gl[lOffset],
                    layoutOTmp, layoutO, layoutUpdate, actualBlockShapePV, nIdx, isLastNTile, qHeadSplitSizeActual,
                    rescaleOPingPongFlag, glFlag, taskPingPongFlag);
#endif
            }

#ifdef __DAV_VEC__
            taskPingPongFlag = 1 - taskPingPongFlag;
#endif
        }

#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

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

        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
#endif
#ifdef __DAV_VEC__

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

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
                (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX ? HEADS_PROCESS_MAX : (COMPUTE_ELE_NUM / embed);
            uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
            uint32_t loopsTotal = batch * loopsPerBatch;

            for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
                uint32_t batchIdx = loopIdx / loopsPerBatch;
                uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;

                uint32_t offsetTiling = tilingHeadSize + tilingParaSize * batchIdx;
                uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
                uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
                uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);

                if (kvSeqlen == 0) {
                    continue;
                }

                uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + 4);
                uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + 5);
                uint64_t oAddr = (uint64_t)((((uint64_t)oAddrHigh32) << 32) | oAddrLow32);
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

                for (uint32_t qSeqIdx = 0; qSeqIdx < qSeqlen; qSeqIdx++) {
                    epilogueMLAFDRescaleO(
                        gO[oAddr + qSeqIdx * qHeads * embed + loopIdxInBatch * headsProcess * embed],
                        gOCoreTmp
                            [oFdOffset * maxKvSplitCoreNum + qSeqIdx * qHeads * embed * maxKvSplitCoreNum +
                             loopIdxInBatch * headsProcess * maxKvSplitCoreNum * embed],
                        gl[lOffset + qSeqIdx * qHeads * maxKvSplitCoreNum +
                           loopIdxInBatch * headsProcess * maxKvSplitCoreNum],
                        actualHeads, headsProcess, embed, kvSplitCoreNum);
                }
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
CATLASS_GLOBAL void MLA(
    uint64_t hardwareSyncAddr, GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope, GM_ADDR blockTables, GM_ADDR o,
    GM_ADDR s, GM_ADDR p, GM_ADDR oTmp, GM_ADDR oUpdate, GM_ADDR oCoreTmp, GM_ADDR l, GM_ADDR tiling)
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
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    constexpr uint32_t ComputeEleNum = 6144;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel =
        MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax, EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};

    // call kernel
    MLAKernel mla;
    mla(params);
}
