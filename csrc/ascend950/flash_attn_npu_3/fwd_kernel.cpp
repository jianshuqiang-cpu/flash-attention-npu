/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Modified by Minghua Shen, 2026.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_HPP
#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_QK_DN_HPP
#define GEMM_BLOCK_BLOCK_MMAD_FLASH_ATTENTION_PV_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "fwd_resource.hpp"
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_SOFTMAX_HIGH_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_FLASH_ATTENTION_RESCALE_O_HPP
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#include "kernel_common.hpp"
#include "fa_block.h"
#include "rescale_o.hpp"
#include "online_softmax.hpp"
#include "init_outputs.hpp"
#include "qk_matmul.hpp"
#include "qk_matmul_DN.hpp"
#include "pv_matmul.hpp"
#include "tilingdata.h"
#include "fd_combine.hpp"
#include "paired_decode_kv.hpp"
#include "resident_kv_pages.hpp"

using namespace Catlass;
using namespace tla;

static constexpr uint32_t PRE_LAUNCH = 2;
static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;
static constexpr uint32_t UB_S_BUF_STAGES = 2;
static constexpr uint32_t RESIDENT_KEY_BANKS = 4;

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO, Format qFormat,
          Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape, MaskCategory maskCategory,
          CacheLayout cacheLayout, bool IsDN, bool LseMode, bool MixedKv = false, bool PairedHeads = false,
          bool SingleQueryWindow = false, bool PairedKv = false, bool GroupedHeads = false,
          bool ResidentPages = false>
class FAIKernel950 {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;

    using ElementQ = std::conditional_t<IsDN, typename BlockMmadQK::ElementB, typename BlockMmadQK::ElementA>;
    using ElementK = std::conditional_t<IsDN, typename BlockMmadQK::ElementA, typename BlockMmadQK::ElementB>;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;

    using LayoutQ = std::conditional_t<IsDN, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<IsDN, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    uint32_t otmpStages_ = 2U;
    bool hasSeqUsedQ = false;
    bool hasSeqUsedKv = false;

    __aicore__ inline FAIKernel950(AscendC::TPipe& pipe) : resource(pipe) {}

    // *Used* lengths from seqused when provided; cu stays the region layout.
    __aicore__ inline int64_t GetQSeqLen(uint32_t bIdx, uint32_t maxQSeqlen, AscendC::GlobalTensor<int32_t>& gQSeq,
                                         AscendC::GlobalTensor<int32_t>& gSeqUsedQ) const
    {
        if constexpr (qFormat == Format::TND) {
            if (hasSeqUsedQ) {
                return static_cast<int64_t>(gSeqUsedQ.GetValue(bIdx));
            }
            return static_cast<int64_t>(gQSeq.GetValue(bIdx + 1) - gQSeq.GetValue(bIdx));
        }
        return static_cast<int64_t>(maxQSeqlen);
    }

    __aicore__ inline int64_t GetKvSeqLen(uint32_t bIdx, AscendC::GlobalTensor<int32_t>& gKvSeq,
                                          AscendC::GlobalTensor<int32_t>& gSeqUsedKv) const
    {
        if constexpr (kvFormat == Format::BNSD) {
            // BNSD KV is indexed per (batch, head) with the allocated plane length;
            // never use the TND cumulative-length differencing.
            if (hasSeqUsedKv) {
                return static_cast<int64_t>(gSeqUsedKv.GetValue(bIdx));
            }
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        } else if constexpr (kvFormat == Format::TND) {
            if (hasSeqUsedKv) {
                return static_cast<int64_t>(gSeqUsedKv.GetValue(bIdx));
            }
            if constexpr (kvcacheType == CacheMode::normalCache) {
                return static_cast<int64_t>(gKvSeq.GetValue(bIdx + 1) - gKvSeq.GetValue(bIdx));
            }
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        } else {
            return static_cast<int64_t>(gKvSeq.GetValue(bIdx));
        }
    }

#ifdef __DAV_VEC__
    __aicore__ inline void PrefetchResidentKv(fa950_mixed::KvDequantBuffered& dequant,
        const FAIKernelParams& params, AscendC::GlobalTensor<int32_t> pages, uint32_t head, uint32_t page, bool keyLane)
    {
        dequant.Prefetch(keyLane ? params.k : params.v, keyLane ? params.kDescale : params.vDescale,
            pages, head, kvHeads_, keyLane ? params.storage.keyPageStride : params.storage.valuePageStride,
            keyLane ? params.storage.keyTokenStride : params.storage.valueTokenStride,
            keyLane ? params.storage.keyHeadStride : params.storage.valueHeadStride, page, 128U, keyLane ? 1U : 0U);
    }

    __aicore__ inline void ProduceResidentKv(fa950_mixed::KvDequantBuffered& dequant, uint32_t bank,
        uint32_t pvL1AddrStart, bool keyLane)
    {
        constexpr uint32_t pageBytes = 128U * 192U * sizeof(bfloat16_t);
        const uint32_t base = keyLane ? qL1BufNum_ * pageBytes :
            pvL1AddrStart + pL1BufNum_ * 128U * 128U * sizeof(ElementP);
        dequant.Produce(resource, 128U, base + bank * pageBytes, (keyLane ? 7U : 9U) + bank,
                        keyLane ? 1U : 0U);
    }

    __aicore__ inline void ProducePairedKv(fa950_mixed::KvDequantBuffered& halfDequant, const FAIKernelParams& params,
                                           AscendC::GlobalTensor<int32_t> pages, uint32_t kvHeadIdx,
                                           uint32_t kvSplitStart, uint32_t kvSplitEnd, uint32_t kvSLoopNum,
                                           uint32_t noSkipKvS, uint32_t pvL1AddrStart, uint32_t step)
    {
        const bool keyLane = AscendC::GetSubBlockIdx() == 0U;
        const uint32_t kvSlot = keyLane ? 1U : 0U;
        const auto cache = keyLane ? params.k : params.v;
        const auto scales = keyLane ? params.kDescale : params.vDescale;
        const auto pageStride = keyLane ? params.storage.keyPageStride : params.storage.valuePageStride;
        const auto tokenStride = keyLane ? params.storage.keyTokenStride : params.storage.valueTokenStride;
        const auto headStride = keyLane ? params.storage.keyHeadStride : params.storage.valueHeadStride;
        if (step == 0U) {
            const uint32_t rows = kvSplitStart + 1U == kvSLoopNum ? noSkipKvS - kvSplitStart * 128U : 128U;
            halfDequant.Prefetch(cache, scales, pages, kvHeadIdx, kvHeads_,
                pageStride, tokenStride, headStride, kvSplitStart, rows, kvSlot);
        }
        // Keep the first two keys on the original schedule for startup.
        // In steady state, prepare 64 rows during each query-head step.
        const bool wholeKey = keyLane && step < 2U;
        const uint32_t firstHalf = wholeKey ? 0U : step % 2U;
        const uint32_t endHalf = wholeKey ? 2U : firstHalf + 1U;
        const uint32_t page = kvSplitStart + step / 2U + (keyLane && step != 0U ? 1U : 0U);
        if (page < kvSplitEnd) {
            const uint32_t rows = page + 1U == kvSLoopNum ? noSkipKvS - page * 128U : 128U;
            // Cube consumes the first visible page in bank zero.
            // SWA can start at an odd physical tile index.
            const uint32_t bank = maskCategory == MaskCategory::MASK_SWA ?
                (page - kvSplitStart) % 2U : page % 2U;
            const uint32_t kOffset = qkL1TileM_ * qkL1TileKLeft_ * 2U * qL1BufNum_
                + bank * qkL1TileKRight_ * qkL1TileN_ * 2U;
            const uint32_t vOffset = pvL1AddrStart + pvL1TileM_ * pvL1TileKLeft_ * 2U * pL1BufNum_
                + bank * pvL1TileKRight_ * pvL1TileN_ * 2U;
            halfDequant.Produce(resource, rows, keyLane ? kOffset : vOffset,
                (keyLane ? 7U : 9U) + bank, kvSlot, firstHalf, endHalf);
            if (endHalf == 2U && page + 1U < kvSplitEnd) {
                const uint32_t next = page + 1U;
                const uint32_t nextRows = next + 1U == kvSLoopNum ? noSkipKvS - next * 128U : 128U;
                halfDequant.Prefetch(cache, scales, pages, kvHeadIdx, kvHeads_,
                    pageStride, tokenStride, headStride, next, nextRows, kvSlot);
            }
        }
    }
#endif

    __aicore__ inline void operator()(FAIKernelParams const& params)
    {
        __gm__ FAInferTilingData* faiTilingData = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gSeqUsedQ;
        gSeqUsedQ.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedQ);
        AscendC::GlobalTensor<int32_t> gSeqUsedKv;
        gSeqUsedKv.SetGlobalBuffer((__gm__ int32_t*)params.seqUsedKv);
        hasSeqUsedQ = params.seqUsedQ != nullptr;
        hasSeqUsedKv = params.seqUsedKv != nullptr;
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTables));
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        if constexpr (ResidentPages || (PairedHeads && !MixedKv)) {
            gO.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float*)params.lse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float*)(params.workSpace + faiTilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float*)(params.workSpace + faiTilingData->fdPartialOOffset));
        //tiling data

        batch_ = faiTilingData->batch;
        qHeads_ = faiTilingData->numHeads;
        kvHeads_ = faiTilingData->kvHeads;
        embed_ = faiTilingData->embeddingSize;
        embedV_ = faiTilingData->embeddingSizeV;
        firstBatchTaskNum_ = faiTilingData->firstBatchTaskNum;
        totalTaskNum_ = faiTilingData->totalTaskNum;
        scaleValue_ = params.scaleValue;
        softcapValue_ = params.softcapValue;
        // base tile info
        qBaseTile_ = faiTilingData->qBaseTile;
        kvBaseTile_ = faiTilingData->kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = faiTilingData->actSeqAval;
        // aligned seqlen q & kv
        qSeqlenAligned_ = faiTilingData->qSeqlenAligned;
        kvSeqlenAligned_ = faiTilingData->kvSeqlenAligned;
        maxNumBlocksPerBatch_ = params.maxNumBlocksPerBatch;
        blockSize_ = faiTilingData->blockSize;
        numBlocks_ = params.numBlocks;
        qkL1TileM_ = faiTilingData->qkL1TileM;
        qkL1TileN_ = faiTilingData->qkL1TileN;
        qkL1TileKLeft_ = faiTilingData->qkL1TileKLeft;
        qkL1TileKRight_ = faiTilingData->qkL1TileKRight;
        pvL1TileM_ = faiTilingData->pvL1TileM;
        pvL1TileN_ = faiTilingData->pvL1TileN;
        pvL1TileKLeft_ = faiTilingData->pvL1TileKLeft;
        pvL1TileKRight_ = faiTilingData->pvL1TileKRight;
        constexpr bool nativePair = PairedHeads && !MixedKv;
        const uint32_t groupSize = qHeads_ / kvHeads_;
        const uint32_t groupedTasksPerKvHead = (groupSize + 1U) / 2U;
        const uint32_t groupedTasksPerQTile = groupedTasksPerKvHead * kvHeads_;
        // Native metadata is shared with ordinary FP16/BF16 callers. Only the paired
        // specialization reserves the second Q bank in the per-core L1 layout.
        qL1BufNum_ = nativePair ? 2U : faiTilingData->qL1BufNum;
        kL1BufNum_ = faiTilingData->kL1BufNum;
        vL1BufNum_ = faiTilingData->vL1BufNum;
        pL1BufNum_ = faiTilingData->pL1BufNum;
        windowSizeLeft_ = faiTilingData->windowSizeLeft;
        windowSizeRight_ = faiTilingData->windowSizeRight;
        flashDecodeFlag_ = faiTilingData->flashDecodeFlag;
        fdCombineTaskNum_ = faiTilingData->fdCombineTaskNum;
        fdRowCapacity_ = faiTilingData->fdRowCapacity;
        fdLseSubStride_ = faiTilingData->fdLseSubStride;

        // Prefetch K and V pages within the current paired SWA task.
        constexpr bool prefetchNativeKv = nativePair && maskCategory == MaskCategory::MASK_SWA;
        constexpr bool compactPipeline = ResidentPages || prefetchNativeKv;
        constexpr bool leadKeyPages = prefetchNativeKv && kvcacheShape == PageShape::BnBsND;
        if (leadKeyPages) {
            kL1BufNum_ = 3U;
        }
        if (prefetchNativeKv) {
            // Use single-tile K slots so three V slots fit in L1.
            qkL1TileN_ = 128U;
            vL1BufNum_ = 3U;
            pL1BufNum_ = 2U;
        }
        static_assert(8U * 128U * 192U * sizeof(half) + 2U * 128U * 128U * sizeof(half) <=
                      fa950_plan::L1_CAPACITY, "Two Q and three K/V slots fit beside two P slots");

        if constexpr (ResidentPages) {
            qkL1TileN_ = 128U;
            kL1BufNum_ = RESIDENT_KEY_BANKS;
            vL1BufNum_ = fa950_resident::BANKS;
            pL1BufNum_ = 2U;
        }
        static_assert((2U + RESIDENT_KEY_BANKS + fa950_resident::BANKS) * 128U * 192U * sizeof(bfloat16_t) +
                      2U * 128U * 128U * sizeof(bfloat16_t) <= fa950_plan::L1_CAPACITY,
                      "Resident prefill needs four K, three V, two Q and two P banks");

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        auto pvL1AddrStart_ = qkL1TileM_ * qkL1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                              qkL1TileKRight_ * qkL1TileN_ * kL1BufNum_ * sizeof(ElementK);
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * i);
        }
        const bool mixedKv = !nativePair && (MixedKv || params.kDescale != nullptr);
        static_assert(!PairedHeads || (!IsDN && (MixedKv ||
                      ((std::is_same_v<ElementQ, half> || std::is_same_v<ElementQ, bfloat16_t>) &&
                       qFormat == Format::TND && kvcacheType == CacheMode::pagedCache))),
                      "Paired heads require mixed KV or native paged FP16/BF16");
        static_assert(!GroupedHeads || nativePair, "Grouped tasks require native paired prefill");
        const bool bufferedKv = !PairedHeads && mixedKv && qBaseTile_ <= 64U;
        const bool splitRowKv = mixedKv && !bufferedKv && !IsDN;
        const bool twoKvOutputs = bufferedKv &&
            static_cast<uint64_t>(faiTilingData->maxQSeqlen) * (qHeads_ / kvHeads_) <= 32U;
        const bool earlyKv = twoKvOutputs || (bufferedKv && !IsDN);
        const uint32_t kvPrefetchDistance = earlyKv ? 2U : 1U;
        // GetQNBlockTile limits merged rows to 64 for this path. Fixpipe splits
        // them between two AIVs, so LO/GO need at most 32 rows per AIV.
        // If the entire Q/GQA group has <=32 rows, 16 per AIV suffices.
        const uint32_t outputRows = twoKvOutputs ? 16U : (bufferedKv ? 32U : 64U);
        const uint32_t scoreRows = earlyKv ? outputRows : 64U;
        const auto ubPlan = PairedKv ? fa950_plan::MakeUbPlan(embedV_, 2U, 16U, 16U) :
            nativePair ? fa950_plan::MakeUbPlan(embedV_, 1U, 64U, 64U, 1U, 1U) :
            mixedKv ? fa950_plan::MakeUbPlan(embedV_, 1U, outputRows, scoreRows, splitRowKv ? 1U : 2U, PairedHeads ? 1U : 2U) :
                      fa950_plan::KernelUbPlan(embedV_, qBaseTile_);
        static_assert(fa950_plan::MakeUbPlan(192, 1, 64, 64, 1, 1).total +
                      64U * 192U * sizeof(float) + 2U * 64U * sizeof(float) <= fa950_plan::UB_CAPACITY,
                      "Native pair must fit a second output and its softmax statistics");
        static_assert(fa950_plan::L1Bytes(192, 192) + 128U * 192U * sizeof(half) <= fa950_plan::L1_CAPACITY,
                      "Native pair must fit two Q banks alongside K, V and P");
        otmpStages_ = ubPlan.otmpStages;
        for (uint32_t i = 0; i < UB_S_BUF_STAGES; ++i) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubPlan.s + ubPlan.sBytes * i);
        }
        for (uint32_t i = 0; i < otmpStages_; ++i) {
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubPlan.lo + ubPlan.loBytes * i);
        }
        const uint32_t ubSRoundTile = 128;

        Gemm::Block::BlockMmadQKTileHelper qkL1TileHelper(qkL1TileM_, qkL1TileN_, qkL1TileKLeft_, qkL1TileKRight_,
                                                          qL1BufNum_, kL1BufNum_);
        Gemm::Block::BlockMmadPVTileHelper pvL1TileHelper(pvL1TileM_, pvL1TileN_, pvL1TileKLeft_, pvL1TileKRight_,
                                                          pL1BufNum_, vL1BufNum_);

        qkL0ATotalStages_ = CeilDiv(Q_M_TILE_MAX, BlockMmadQK::L0_TILE_M) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        qkL0BTotalStages_ = CeilDiv(kvBaseTile_, BlockMmadQK::L0_TILE_N) * CeilDiv(embed_, BlockMmadQK::L0_TILE_K);
        pvL0ATotalStages_ =
            CeilDiv(Q_M_TILE_MAX, BlockMmadPV::L0_TILE_M) * CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K);
        pvL0BTotalStages_ = CeilDiv(kvBaseTile_, BlockMmadPV::L0_TILE_K) * CeilDiv(embedV_, BlockMmadPV::L0_TILE_N);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags<4, 4, 4>(compactPipeline);
#ifdef __DAV_CUBE__
        if constexpr (!IsDN && nativePair) {
            if (leadKeyPages) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BlockMmadQK::THIRD_K_BANK_EVENT);
            }
        }
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, qkL1TileHelper);
        BlockMmadPV blockMmadPV(resource, pvL1AddrStart_, pvL1TileHelper);
        if constexpr (!IsDN) {
            blockMmadQK.externalKv = mixedKv;
            blockMmadQK.pairedHeads = PairedHeads && !PairedKv;
            blockMmadQK.singleKvProducer = bufferedKv || splitRowKv;
        }
        blockMmadPV.externalKv = mixedKv;
        blockMmadPV.prefetchedV = prefetchNativeKv;
        blockMmadPV.pairedHeads = PairedHeads && !PairedKv;
        blockMmadPV.singleKvProducer = bufferedKv || splitRowKv;
        if constexpr (!IsDN) {
            blockMmadQK.useUnitFix = compactPipeline;
            blockMmadQK.leadNativeK = leadKeyPages;
        }
        blockMmadPV.useUnitFix = compactPipeline;
        if (!ResidentPages && mixedKv) {
            for (uint32_t flag = 7; flag < 11; ++flag) {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(flag);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(flag + 16);
            }
        }
        blockMmadQK.SetPageStride(params.storage.keyPageStride);
        blockMmadPV.SetPageStride(params.storage.valuePageStride);
#endif

#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        auto secondPlan = ubPlan;
        if (PairedHeads) {
            secondPlan.go = ubPlan.total;
            secondPlan.gm = secondPlan.go + ubPlan.loBytes;
            secondPlan.gl = secondPlan.gm + 64U * sizeof(float);
            secondPlan.total = secondPlan.gl + 64U * sizeof(float);
        }
        EpilogueOnlineSoftmax firstSm(resource, ubPlan, scaleValue_, softcapValue_);
        EpilogueOnlineSoftmax secondSm(resource, secondPlan, scaleValue_, softcapValue_);
        const bool packFinalOutput = (MixedKv && PairedHeads && !PairedKv &&
            maskCategory == MaskCategory::MASK_SWA && qBaseTile_ == 128U && embedV_ == 192U) ||
            prefetchNativeKv || (nativePair && std::is_same_v<ElementQ, half> &&
                maskCategory == MaskCategory::MASK_SWA && qBaseTile_ == 128U && embedV_ == 192U);
        EpilogueRescaleO firstO(resource, ubPlan, mixedKv && !PairedHeads, packFinalOutput);
        EpilogueRescaleO secondO(resource, secondPlan, mixedKv && !PairedHeads, packFinalOutput);
        fa950_mixed::KvDequant kvDequant(resource, mixedKv ? ubPlan.total : 0U);
        fa950_mixed::KvDequantBuffered halfDequant(resource, splitRowKv ? secondPlan.total : 0U);
        fa950_mixed::KvDequantWhole bufferedDequant(resource, bufferedKv ? ubPlan.total : 0U, earlyKv, twoKvOutputs);
        Epilogue::Block::InitOutputs950<ArchTag, ElementO> initOutputs(resource);
#endif

        int64_t strideQ = 0;
        int64_t strideO = 0;
        int64_t strideK = 0;
        int64_t strideV = 0;
        // Per-head element distances. For BSND the head plane is D wide and the
        // token stride is H*D; for BNSD each head plane is S_alloc*D wide and
        // the token stride is D.
        int64_t qHeadStride = 0;
        int64_t oHeadStride = 0;
        int64_t kvHeadStrideK = 0;
        int64_t kvHeadStrideV = 0;
        int64_t kvNumTokens;
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvNumTokens = gActualKvseqlen.GetValue(batch_);
        } else {
            kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1);
        }
        // int64_t kvNumTokens = gActualKvseqlen.GetValue(batch_ - 1); // used for TND_NZ

        if constexpr (qFormat == Format::BNSD) {
            strideQ = embed_;
            qHeadStride = static_cast<int64_t>(faiTilingData->maxQSeqlen) * embed_;
        } else {
            strideQ = qHeads_ * embed_;
            qHeadStride = embed_;
        }
        // Q/K/V are read with the input layout's strides (BNSD stays zero-copy),
        // but O is always addressed with BSND semantics so the returned tensor is
        // a contiguous (B, S, H, D) block regardless of the input layout. The
        // head plane is D wide and the token stride is H*D in BSND.
        strideO = qHeads_ * embedV_;
        oHeadStride = embedV_;
        kvHeadStrideK = embed_;
        kvHeadStrideV = embedV_;
        if constexpr (cacheLayout == CacheLayout::nd) {
            if constexpr (kvFormat == Format::BNSD) {
                strideK = embed_;
                strideV = embedV_;
                kvHeadStrideK = static_cast<int64_t>(params.kvSeqlenAllocated) * embed_;
                kvHeadStrideV = static_cast<int64_t>(params.kvSeqlenAllocated) * embedV_;
            } else {
                strideK = kvHeads_ * embed_;
                strideV = kvHeads_ * embedV_;
            }
        } else {
            strideK = 16;
            strideV = 16;
        }

        if constexpr (kvcacheType == CacheMode::pagedCache && cacheLayout == CacheLayout::nd) {
            strideK = params.storage.keyTokenStride;
            strideV = params.storage.valueTokenStride;
            kvHeadStrideK = params.storage.keyHeadStride;
            kvHeadStrideV = params.storage.valueHeadStride;
        }

        uint32_t embedVRound = RoundUp(embedV_, 16);
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t blockBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t cachedBatch = 0xFFFFFFFFU;
        // Used lengths drive task counts; physical (region) lengths drive offsets.
        int64_t qSeqlen = GetQSeqLen(curBatch, faiTilingData->maxQSeqlen, gActualQseqlen, gSeqUsedQ);
        int64_t qPhysicalSeqlen = faiTilingData->maxQSeqlen;
        int64_t kvSeqlen = 0;
        int64_t kvPhysicalSeqlen = gActualKvseqlen.GetValue(curBatch);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        if constexpr (qFormat == Format::TND) {
            qPhysicalSeqlen =
                static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
        }
        if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
            kvPhysicalSeqlen =
                static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
        } else if constexpr (kvFormat == Format::BSND && kvcacheType == CacheMode::normalCache) {
            // Dense BSND is [B, S_alloc, H, D]; S_alloc may exceed cache_seqlens.
            kvPhysicalSeqlen = params.kvSeqlenAllocated;
        }
        if constexpr (MixedKv && !PairedHeads && qFormat == Format::TND &&
                      kvcacheType == CacheMode::pagedCache && maskCategory != MaskCategory::MASK_SWA) {
            if (flashDecodeFlag_ && faiTilingData->fdActiveCoreNum == 32U && batch_ == 1U &&
                faiTilingData->fdBaseTaskNum == kvHeads_ && params.storage.keyTokenStride > embed_) {
                // BBND places adjacent heads next to each other in each token. Assign
                // neighbouring core groups to different heads of the same KV split.
                // The logical index also selects the original partial slot for the merge.
                if (kvHeads_ == 4U) {
                    coreIdx = (coreIdx % 4U) * 8U + coreIdx / 4U;
                } else if (kvHeads_ == 2U) {
                    coreIdx = (coreIdx % 2U) * 16U + coreIdx / 2U;
                }
            }
        }
        uint32_t taskBegin = coreIdx;
        uint32_t taskEnd = totalTaskNum_;
        uint32_t taskStep = coreNum;
        if constexpr (GroupedHeads) {
            // Native tiling counts one task per head for a full Q tile.
            // Convert those counts into the per-group pair/tail grid.
            curTotalTaskNum = firstBatchTaskNum_ / qHeads_ * groupedTasksPerQTile;
            taskEnd = totalTaskNum_ / qHeads_ * groupedTasksPerQTile;
        }
        int32_t fdBaseTaskStart = -1;
        int32_t fdBaseTaskEnd = -1;
        int32_t fdFirstKvTileStart = -1;
        int32_t fdLastKvTileEnd = -1;
        if (!PairedKv && flashDecodeFlag_ != 0U) {
            fdBaseTaskStart = faiTilingData->fdDecodeSchedules[coreIdx].baseTaskStart;
            fdBaseTaskEnd = faiTilingData->fdDecodeSchedules[coreIdx].baseTaskEnd;
            fdFirstKvTileStart = faiTilingData->fdDecodeSchedules[coreIdx].firstKvTileStart;
            fdLastKvTileEnd = faiTilingData->fdDecodeSchedules[coreIdx].lastKvTileEnd;
            taskBegin = static_cast<uint32_t>(fdBaseTaskStart);
            taskEnd = static_cast<uint32_t>(fdBaseTaskEnd);
            taskStep = 1U;
        }
        const bool flashDecode = flashDecodeFlag_ != 0U;
        fa950_resident::PageBanks residentValues;
        uint32_t residentKeyBatch = fa950_resident::INVALID;
        uint32_t residentKeyEnd = 0U;
#ifdef __DAV_CUBE__
        uint32_t nativePageBatch = batch_;
        uint32_t nativePageHead = 0U;
        uint32_t nativeLastPage = fa950_resident::INVALID;
#endif
        uint32_t pipelineTileBase = 0U;
        uint64_t pipelineL0AStageBase = 0U;
        uint64_t pipelineL0BStageBase = 0U;
        uint32_t splitsPerHead = 0U;
        uint32_t kvSplit = 0U;
        if constexpr (PairedKv) {
            const uint32_t headPairs = kvHeads_ / 2U;
            splitsPerHead = faiTilingData->fdActiveCoreNum / headPairs;
            kvSplit = coreIdx / headPairs;
            taskBegin = 2U * (coreIdx % headPairs);
            taskEnd = taskBegin + 1U;
        } else if (PairedHeads && !GroupedHeads) {
            taskBegin *= 2U;
            taskStep *= 2U;
        }
        constexpr bool supportsTaskOrder = MixedKv && !PairedKv && !PairedHeads &&
            (qFormat == Format::TND || qFormat == Format::BSND) && kvcacheType == CacheMode::pagedCache &&
            maskCategory != MaskCategory::MASK_SWA;
        const uint32_t gridSplits = supportsTaskOrder && flashDecodeFlag_ == FD_UNIFORM_SPLIT_GRID ?
            faiTilingData->fdPartialTaskNum / totalTaskNum_ : 0U;
        const bool indexedTasks = supportsTaskOrder && (flashDecodeFlag_ == FD_WHOLE_TASK_INDEXED ||
            (gridSplits != 0U && fdBaseTaskStart >= 0));
        const uint32_t workCount = gridSplits != 0U ? faiTilingData->fdPartialTaskNum : totalTaskNum_;
        if (gridSplits != 0U && !indexedTasks) {
            taskBegin = coreIdx;
            taskEnd = workCount;
            taskStep = coreNum;
        }
        uint32_t chunkShift = 0U;
        if (indexedTasks) {
            chunkShift = faiTilingData->fdWholeTaskOrder.chunkShift;
            taskBegin <<= chunkShift;
            taskEnd <<= chunkShift;
        }
        for (uint32_t base = taskBegin; base < taskEnd; base += taskStep) {
            uint32_t workId = base;
            if (indexedTasks) {
                const uint32_t chunk = faiTilingData->fdWholeTaskOrder.chunkIds[base >> chunkShift];
                workId = (chunk << chunkShift) + (base & ((1U << chunkShift) - 1U));
                if (workId >= workCount) {
                    continue; // Last logical chunk may be shorter.
                }
            }
            const uint32_t taskIdx = gridSplits != 0U ? workId / gridSplits : workId;
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                // Batch strides are recomputed on every batch advance because packed
                // TND regions have a per-batch physical seqlen. Dense BSND uses the
                // allocated S (constant). BNSD planes are maxSeqlen sized so they
                // are constant.
                int64_t qBatchStride = 0;
                if constexpr (qFormat == Format::BNSD) {
                    qBatchStride = static_cast<int64_t>(qHeads_) * qHeadStride;
                } else {
                    qBatchStride = qPhysicalSeqlen * strideQ;
                }
                // O always follows BSND: one batch is qPhysicalSeqlen rows of
                // strideO elements each.
                int64_t oBatchStride = qPhysicalSeqlen * strideO;
                int64_t kBatchStride = 0;
                int64_t vBatchStride = 0;
                if constexpr (kvFormat == Format::BNSD) {
                    kBatchStride = static_cast<int64_t>(kvHeads_) * kvHeadStrideK;
                    vBatchStride = static_cast<int64_t>(kvHeads_) * kvHeadStrideV;
                } else {
                    kBatchStride = kvPhysicalSeqlen * strideK;
                    vBatchStride = kvPhysicalSeqlen * strideV;
                }
                qBOffset += qBatchStride;
                if constexpr (kvcacheType == CacheMode::normalCache) {
                    kBOffset += static_cast<uint64_t>(kBatchStride);
                    vBOffset += static_cast<uint64_t>(vBatchStride);
                } else {
                    blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch_);
                }
                oBOffset += oBatchStride;

                qSeqlen = GetQSeqLen(curBatch, faiTilingData->maxQSeqlen, gActualQseqlen, gSeqUsedQ);
                qPhysicalSeqlen = faiTilingData->maxQSeqlen;
                kvPhysicalSeqlen = gActualKvseqlen.GetValue(curBatch);
                if constexpr (qFormat == Format::TND) {
                    qPhysicalSeqlen =
                        static_cast<int64_t>(gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
                }
                if constexpr (kvFormat == Format::TND && kvcacheType == CacheMode::normalCache) {
                    kvPhysicalSeqlen = static_cast<int64_t>(gActualKvseqlen.GetValue(curBatch + 1) -
                                                            gActualKvseqlen.GetValue(curBatch));
                } else if constexpr (kvFormat == Format::BSND && kvcacheType == CacheMode::normalCache) {
                    kvPhysicalSeqlen = params.kvSeqlenAllocated;
                }
                // GetQSBlockTile is a constant independent of kvSeqlen so the kv
                // length is resolved in the cachedBatch gate below, not here. If this
                // function ever becomes kv dependent move the kv resolution back.
                uint32_t curQNBlockTile =
                    GetQNBlockTile(static_cast<uint32_t>(qSeqlen), groupSize, qBaseTile_ < Q_M_TILE_MAX);
                uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile);
                const uint32_t qNTaskNum = GroupedHeads ? groupedTasksPerQTile : qNBlockNumPerGroup * kvHeads_;
                curTotalTaskNum += (qSeqlen + qBaseTile_ - 1) / qBaseTile_ * qNTaskNum;
            }
            if (curBatch != cachedBatch) {
                kvSeqlen = GetKvSeqLen(curBatch, gActualKvseqlen, gSeqUsedKv);
                cachedBatch = curBatch;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSTileIdx;
            uint32_t kvHeadIdx;
            uint32_t qNStartIdx;
            uint32_t qNBlockSize;
            uint32_t curQNBlockTile =
                GetQNBlockTile(static_cast<uint32_t>(qSeqlen), groupSize, qBaseTile_ < Q_M_TILE_MAX);
            const uint32_t qNBlockNumPerGroup = GroupedHeads ? groupedTasksPerKvHead : CeilDiv(groupSize, curQNBlockTile);
            uint32_t qNTaskNum = qNBlockNumPerGroup * kvHeads_;
            qSTileIdx = taskIdxCurBatch / qNTaskNum;
            uint32_t qNBlockIdx = taskIdxCurBatch - qSTileIdx * qNTaskNum;
            if constexpr (ResidentPages) {
                const uint32_t lanesPerPair = coreNum / (qHeads_ / 2U);
                const uint32_t tilesPerLane = static_cast<uint32_t>(qSeqlen) / qBaseTile_ / lanesPerPair;
                qSTileIdx = (qSTileIdx % lanesPerPair) * tilesPerLane + qSTileIdx / lanesPerPair;
            }
            if constexpr (PairedHeads && maskCategory == MaskCategory::MASK_CAUSAL) {
                // Consecutive waves visit opposite ends of their Q-tile ranges.
                // Keep the head pair and every incomplete wave at its original index.
                const uint32_t tilesPerWave = qNTaskNum == coreNum ? 2U : (2U * qNTaskNum == coreNum ? 4U : 1U);
                const uint32_t wave = qSTileIdx / tilesPerWave;
                const uint32_t waveStart = wave * tilesPerWave;
                if (tilesPerWave > 1U && ((wave + curBatch) & 1U) &&
                    waveStart + tilesPerWave <= static_cast<uint32_t>(qSeqlen) / qBaseTile_) {
                    qSTileIdx = waveStart + (tilesPerWave - 1U - (qSTileIdx - waveStart));
                }
            }
            const uint32_t qNBlockIdxInGroup = qNBlockIdx % qNBlockNumPerGroup;
            kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup;
            const uint32_t qHeadInGroup = qNBlockIdxInGroup * (GroupedHeads ? 2U : curQNBlockTile);
            qNStartIdx = kvHeadIdx * groupSize + qHeadInGroup;
            qNBlockSize = GroupedHeads ? 1U : AscendC::Std::min(curQNBlockTile, groupSize - qHeadInGroup);
            const bool paired = PairedHeads && (!GroupedHeads || qHeadInGroup + 1U < groupSize);
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t qSOffset = qSTileIdx * qBaseTile_;
            gmOffsetQ = qBOffset + qSOffset * strideQ + qNStartIdx * qHeadStride;
            gmOffsetO = oBOffset + qSOffset * strideO + qNStartIdx * oHeadStride;
            uint32_t lseHeadStride = static_cast<uint32_t>(faiTilingData->maxQSeqlen);
            uint64_t lseOffset = static_cast<uint64_t>(qBOffset / strideQ) * qHeads_ +
                                 static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            if constexpr (qFormat == Format::BNSD) {
                // LSE is packed as (B, H, maxQSeqlen); the batch index cannot be
                // recovered from qBOffset/strideQ in BNSD, so use curBatch.
                lseHeadStride = static_cast<uint32_t>(faiTilingData->maxQSeqlen);
                lseOffset = static_cast<uint64_t>(curBatch) * qHeads_ * lseHeadStride +
                            static_cast<uint64_t>(qNStartIdx) * lseHeadStride + qSOffset;
            } else if constexpr (qFormat == Format::TND) {
                lseHeadStride = static_cast<uint32_t>(params.storage.totalQTokens);
                lseOffset = static_cast<uint64_t>(qNStartIdx) * lseHeadStride +
                            static_cast<uint64_t>(qBOffset / strideQ + qSOffset);
            }
            int64_t kvSOffset = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache && kvcacheShape == PageShape::BnNBsD) {
                strideK = embed_;
                strideV = embedV_;
                gmOffsetK = static_cast<uint64_t>(kvHeadIdx) * embed_ * blockSize_;
                gmOffsetV = static_cast<uint64_t>(kvHeadIdx) * embedV_ * blockSize_;
            } else if (cacheLayout == CacheLayout::nd) {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideK;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx) * kvHeadStrideV;
            } else {
                gmOffsetK = kBOffset + static_cast<uint64_t>(kvHeadIdx * embed_) * kvNumTokens;
                gmOffsetV = vBOffset + static_cast<uint64_t>(kvHeadIdx * embedV_) * kvNumTokens;
            }
            uint32_t qsBlockNum = (qSeqlen + qBaseTile_ - 1) / qBaseTile_;
            uint32_t qSBlockSize = qSTileIdx == qsBlockNum - 1 ? qSeqlen - (qsBlockNum - 1) * qBaseTile_ : qBaseTile_;
            uint32_t rowNum = qSBlockSize * qNBlockSize;
            uint32_t rowNumRound = 0;
            if constexpr (IsDN) {
                rowNumRound = RoundUp(rowNum, 32U);
            } else {
                rowNumRound = RoundUp(rowNum, 16U);
            }
            uint32_t qkRowNumRound = rowNumRound;
            uint32_t kvSTileSizeAct = kvBaseTile_;

            uint32_t kvStart = 0;
            uint32_t firstPageSkip = 0;
            uint32_t noSkipKvS = static_cast<uint32_t>(kvSeqlen);
            uint32_t kvSLoopNum = 0;
            int64_t windowSizeLeftStartLen = 0;
            int64_t windowSizeLeftEndLen = 0;
            int64_t windowSizeRightStartLen = 0;
            int64_t windowSizeRightEndLen = 0;
            bool notPreMask = true;
            bool notNextMask = true;
            int32_t delStartRow = 0;
            int32_t delEndRow = static_cast<int32_t>(qSeqlen);
            const int32_t qSTileStart = static_cast<int32_t>(qSTileIdx * qBaseTile_);
            if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                notNextMask = false;
                windowSizeRightStartLen = static_cast<int64_t>(qSTileStart) + kvSeqlen - qSeqlen;
                int64_t causalKvEnd = windowSizeRightStartLen + qSBlockSize;
                if (causalKvEnd < 0) {
                    causalKvEnd = 0;
                }
                noSkipKvS = AscendC::Std::min((uint32_t)kvSeqlen, (uint32_t)causalKvEnd);
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                // Use the actual ragged lengths and signed bounds before clipping.
                // Negative windows are unlimited; Q and K have different bounds.
                const int64_t qLength = static_cast<int64_t>(qSeqlen);
                const int64_t kvLength = static_cast<int64_t>(kvSeqlen);
                const int64_t seqDiff = kvLength - qLength;
                const int64_t qBlockBase = static_cast<int64_t>(qSTileIdx) * qBaseTile_;
                notPreMask = windowSizeLeft_ < 0 || windowSizeLeft_ >= kvLength - 1;
                notNextMask = windowSizeRight_ < 0 || windowSizeRight_ >= qLength - 1;
                if (!notPreMask) {
                    const int64_t leftOrigin = seqDiff - windowSizeLeft_;
                    windowSizeLeftStartLen = qBlockBase + leftOrigin;
                    windowSizeLeftEndLen = windowSizeLeftStartLen + qSBlockSize;
                    const int64_t firstKey =
                        AscendC::Std::min(kvLength, AscendC::Std::max(int64_t{0}, windowSizeLeftStartLen));
                    kvStart = static_cast<uint32_t>(firstKey) / kvBaseTile_;
                    if constexpr (SingleQueryWindow) {
                        firstPageSkip = static_cast<uint32_t>(firstKey) % kvBaseTile_;
                    }
                    if (windowSizeLeftEndLen > kvLength) {
                        delStartRow = static_cast<int32_t>(
                            AscendC::Std::min(qLength, AscendC::Std::max(int64_t{0}, kvLength - leftOrigin)));
                    }
                }
                if (!notNextMask) {
                    const int64_t rightOrigin = seqDiff + windowSizeRight_;
                    windowSizeRightStartLen = qBlockBase + rightOrigin;
                    // Exclusive upper bound for the last Q row in this tile.
                    windowSizeRightEndLen = windowSizeRightStartLen + qSBlockSize;
                    const int64_t endKey =
                        AscendC::Std::min(kvLength, AscendC::Std::max(int64_t{0}, windowSizeRightEndLen));
                    noSkipKvS = static_cast<uint32_t>(endKey);
                    if (windowSizeRightStartLen < 0) {
                        delEndRow = static_cast<int32_t>(
                            AscendC::Std::min(qLength, AscendC::Std::max(int64_t{0}, -rightOrigin)));
                    }
                }
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            } else {
                kvSLoopNum = static_cast<uint32_t>(CeilDiv(noSkipKvS, static_cast<int64_t>(kvBaseTile_)));
            }

            uint32_t fullyMaskedRowsPerHead = 0;
            if constexpr (maskCategory == MaskCategory::MASK_CAUSAL) {
                fullyMaskedRowsPerHead = noSkipKvS < qSBlockSize ? qSBlockSize - noSkipKvS : 0;
            }

            bool emptySpan = (kvSLoopNum == 0);
            if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                emptySpan = emptySpan || (kvStart >= kvSLoopNum);
            }
            if (emptySpan) {
#ifdef __DAV_VEC__
                initOutputs.template operator()<LseMode>(gO[gmOffsetO], gLse[lseOffset], qSBlockSize, qNBlockSize,
                                                         lseHeadStride, embedV_, static_cast<uint32_t>(strideO),
                                                         static_cast<uint32_t>(oHeadStride));
                if (paired) {
                    initOutputs.template operator()<LseMode>(gO[gmOffsetO + qNBlockSize * oHeadStride], gLse[lseOffset + qNBlockSize * lseHeadStride], qSBlockSize, qNBlockSize,
                                                         lseHeadStride, embedV_, static_cast<uint32_t>(strideO),
                                                         static_cast<uint32_t>(oHeadStride));
                }
#endif
                continue;
            }
            uint32_t kvSplitStart = kvStart;
            uint32_t kvSplitEnd = kvSLoopNum;
            bool writePartial = false;
            uint32_t partialId = 0U;
            if constexpr (PairedKv) {
                kvSplitStart = kvSLoopNum * kvSplit / splitsPerHead;
                kvSplitEnd = kvSLoopNum * (kvSplit + 1U) / splitsPerHead;
                partialId = kvHeadIdx * splitsPerHead + kvSplit;
                writePartial = true;
            } else if (gridSplits != 0U) {
                const uint32_t part = workId - taskIdx * gridSplits;
                kvSplitStart = kvSLoopNum * part / gridSplits;
                kvSplitEnd = kvSLoopNum * (part + 1U) / gridSplits;
                partialId = workId;
                writePartial = true;
            } else if (flashDecodeFlag_ != 0U && fdCombineTaskNum_ != 0U) {
                if (taskIdx == static_cast<uint32_t>(fdBaseTaskStart)) {
                    kvSplitStart = static_cast<uint32_t>(fdFirstKvTileStart);
                }
                if (taskIdx + 1U == static_cast<uint32_t>(fdBaseTaskEnd)) {
                    kvSplitEnd = static_cast<uint32_t>(fdLastKvTileEnd);
                }
                kvSplitEnd = AscendC::Std::min(kvSplitEnd, kvSLoopNum);
                kvSplitStart = AscendC::Std::min(kvSplitStart, kvSplitEnd);
                for (uint32_t combineIdx = 0; combineIdx < fdCombineTaskNum_; ++combineIdx) {
                    const int32_t combineBaseTask = faiTilingData->fdCombineSchedules[combineIdx].baseTask;
                    if (combineBaseTask == static_cast<int32_t>(taskIdx)) {
                        writePartial = true;
                        partialId = static_cast<uint32_t>(faiTilingData->fdCombineSchedules[combineIdx].partialStart +
                                                          static_cast<int32_t>(coreIdx) -
                                                          faiTilingData->fdCombineSchedules[combineIdx].firstCore);
                        break;
                    }
                }
            }
            const uint32_t pageCount = kvSplitEnd - kvSplitStart;
            // In each pair, operations 2*i and 2*i+1 consume the same K/V page.
            const uint32_t headCount = paired ? 2U : 1U;
            const uint32_t kvSLoopCount = headCount * pageCount;
            if (kvSLoopCount == 0U) {
                continue;
            }
#ifdef __DAV_CUBE__
            if constexpr (GroupedHeads) {
                blockMmadQK.pairedHeads = paired;
                blockMmadQK.leadNativeK = leadKeyPages && paired;
                blockMmadPV.pairedHeads = paired;
                blockMmadPV.prefetchedV = prefetchNativeKv && paired;
            }
            uint32_t qShapeCol = strideQ;
            uint32_t kShapeCol = strideK;
            uint32_t vShapeCol = strideV;
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, qShapeCol);
            auto gmQLayoutTlaDN = tla::MakeLayout<ElementQ, LayoutQ>(qShapeCol, qBaseTile_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            auto gmQTensorTlaDN = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTlaDN, Arch::PositionGM{});

            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            const bool loadPair = paired && !PairedKv && qFormat == Format::TND &&
                kvcacheType == CacheMode::pagedCache && maskCategory == MaskCategory::MASK_SWA &&
                qNBlockSize == 1U && (qSBlockSize == 128U || (prefetchNativeKv && qSBlockSize == 64U)) &&
                qHeadStride == embed_ && strideQ <= 65535U;
            const bool deferSecondQ = prefetchNativeKv && loadPair;
            if constexpr (IsDN) {
                blockMmadQK.loadQGM(gmQTensorTlaDN, actualBlockShapeQ, qSBlockSize, qNBlockSize, qHeadStride);
            } else {
                blockMmadQK.qBank = 0U;
                blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ, qSBlockSize, qNBlockSize, qHeadStride,
                                    loadPair && !deferSecondQ);
                if (paired && !loadPair) {
                    auto nextQ = tla::MakeTensor(gQ[gmOffsetQ + qNBlockSize * qHeadStride], gmQLayoutTla, Arch::PositionGM{});
                    blockMmadQK.qBank = 1U;
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    blockMmadQK.loadQGM(nextQ, actualBlockShapeQ, qSBlockSize, qNBlockSize, qHeadStride);
                }
            }
            uint32_t kShapeRow = 0;
            if constexpr (kvcacheType == CacheMode::pagedCache) {
                kShapeRow = numBlocks_ * blockSize_;
            } else {
                kShapeRow = kvSeqlen;
            }
            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeCol, kShapeRow);
            auto gmKLayoutTlaDN = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});
            auto gmKTensorTlaDN = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTlaDN, Arch::PositionGM{});
            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});
            const bool reuseFirstPage = leadKeyPages && paired && nativePageBatch == curBatch &&
                nativePageHead == kvHeadIdx && nativeLastPage == kvSplitStart &&
                (kvSplitStart + 1ULL) * kvBaseTile_ <= noSkipKvS;
            if constexpr (!IsDN && leadKeyPages) {
                if (reuseFirstPage) {
                    blockMmadQK.PrefetchNativeKPage(gmKTensorTla, gBlockTable[blockBOffset],
                        kvSplitStart, blockSize_, embed_, noSkipKvS, true);
                }
                nativePageBatch = curBatch;
                nativePageHead = kvHeadIdx;
                nativeLastPage = paired && kvSplitEnd * uint64_t{128} <= noSkipKvS ?
                    kvSplitEnd - 1U : fa950_resident::INVALID;
            }
            if (prefetchNativeKv && paired) {
                blockMmadPV.PrefetchNativeV(gmVTensorTla, gBlockTable[blockBOffset],
                    kvSplitStart, embedV_, noSkipKvS, blockSize_, reuseFirstPage);
            }
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = strideO;
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);

#endif
#ifdef __DAV_VEC__
            if (bufferedKv && kvSLoopCount != 0U) {
                for (uint32_t ahead = 0; ahead < kvPrefetchDistance && ahead < kvSLoopCount; ++ahead) {
                    const uint32_t firstTile = kvSplitStart + ahead;
                    const uint32_t rows = firstTile == kvSLoopNum - 1 ? noSkipKvS - firstTile * 128U : 128U;
                    bufferedDequant.Prefetch(params.v, params.vDescale, gBlockTable[blockBOffset],
                        kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                        params.storage.valueHeadStride, firstTile, rows, 0U, ahead == 0U ? firstPageSkip : 0U);
                    bufferedDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                        kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                        params.storage.keyHeadStride, firstTile, rows, 1U, ahead == 0U ? firstPageSkip : 0U);
                }
            }
            if (splitRowKv && !paired && kvSLoopCount != 0U) {
                const uint32_t rows = kvSplitStart == kvSLoopNum - 1 ? noSkipKvS - kvSplitStart * 128U : 128U;
                halfDequant.Prefetch(params.v, params.vDescale, gBlockTable[blockBOffset],
                    kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                    params.storage.valueHeadStride, kvSplitStart, rows, 0U);
                halfDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                    kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                    params.storage.keyHeadStride, kvSplitStart, rows, 1U);
                const uint32_t seq = flashDecode ? pipelineTileBase : 0U;
                const uint32_t offset = qkL1TileM_ * qkL1TileKLeft_ * 2U * qL1BufNum_
                    + (seq % kL1BufNum_) * qkL1TileKRight_ * qkL1TileN_ * 2U;
                halfDequant.Produce(resource, rows, offset, 7U + seq % kL1BufNum_, 1U);
                if (kvSLoopCount > 1U) {
                    const uint32_t next = kvSplitStart + 1U;
                    const uint32_t nextRows = next == kvSLoopNum - 1 ? noSkipKvS - next * 128U : 128U;
                    halfDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                        kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                        params.storage.keyHeadStride, next, nextRows, 1U);
                }
            }
            // Prime K0 so the loop can convert K(i+1) before Softmax(i).
            // The two L1 K slots still use their existing FREE/READY credits.
            if (earlyKv && kvSLoopCount != 0U) {
                if (kvPrefetchDistance < kvSLoopCount) {
                    const uint32_t tile = kvSplitStart + kvPrefetchDistance;
                    const uint32_t rows = tile == kvSLoopNum - 1 ? noSkipKvS - tile * 128U : 128U;
                    bufferedDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                        kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                        params.storage.keyHeadStride, tile, rows, 1U);
                }
                const uint32_t seq = flashDecode ? pipelineTileBase : 0U;
                const uint32_t rows = kvSplitStart == kvSLoopNum - 1 ? noSkipKvS - kvSplitStart * 128U : 128U;
                const uint32_t offset = qkL1TileM_ * qkL1TileKLeft_ * 2U * qL1BufNum_
                    + (seq % kL1BufNum_) * qkL1TileKRight_ * qkL1TileN_ * 2U;
                bufferedDequant.Produce(resource, rows - firstPageSkip, offset, 7U + seq % kL1BufNum_, 1U);
            }

#endif
            fa950_resident::Window residentWindow{};
            uint32_t firstKeyMiss = 0U;
            if constexpr (ResidentPages) {
                residentValues.Plan(residentWindow, kvSplitStart, kvSplitEnd, curBatch, kvHeadIdx);
                // A worker advances by one Q tile for a fixed head pair.
                // Its last four K pages remain in their logical modulo slots.
                firstKeyMiss = residentKeyBatch == curBatch ? residentKeyEnd : kvSplitStart;
                residentKeyBatch = curBatch;
                residentKeyEnd = kvSplitEnd;
            }
            uint32_t nextKey = 0U;
            if constexpr (ResidentPages) {
                const bool keyMiss = kvSplitStart >= firstKeyMiss;
                const bool valueMiss = residentWindow.Miss(0U);
                const uint32_t keyBank = kvSplitStart % RESIDENT_KEY_BANKS;
                const uint32_t valueBank = residentWindow.banks[0];
                nextKey = AscendC::Std::max(kvSplitStart + 1U, firstKeyMiss);
#ifdef __DAV_CUBE__
                if (keyMiss) {
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(7U + keyBank);
                }
                if (valueMiss) {
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(9U + valueBank + 16U);
                }
#endif
#ifdef __DAV_VEC__
                const bool keyLane = AscendC::GetSubBlockIdx() == 0U;
                if (keyLane ? keyMiss : valueMiss) {
                    PrefetchResidentKv(halfDequant, params, gBlockTable[blockBOffset], kvHeadIdx, kvSplitStart, keyLane);
                    ProduceResidentKv(halfDequant, keyLane ? keyBank : valueBank, pvL1AddrStart_, keyLane);
                }
#endif
            }
            for (uint32_t kvSTileRelIdx = 0; kvSTileRelIdx < kvSLoopCount + PRE_LAUNCH; ++kvSTileRelIdx) {
                const uint32_t nextPage = kvSTileRelIdx / 2U + 1U;
                const bool firstHeadStep = (kvSTileRelIdx & 1U) == 0U;
                const bool nextMiss = ResidentPages && nextPage < pageCount && residentWindow.Miss(nextPage);
                const bool lateNext = nextMiss && nextPage + 1U == pageCount && residentWindow.lateTail;
                const bool retireTailPredecessor = ResidentPages && residentWindow.lateTail &&
                    kvSTileRelIdx + 1U == kvSLoopCount;
                const bool fetchKey = ResidentPages && firstHeadStep && nextKey < kvSplitEnd &&
                    nextKey <= kvSplitStart + kvSTileRelIdx / 2U + RESIDENT_KEY_BANKS - 1U;
                if constexpr (ResidentPages) {
#ifdef __DAV_CUBE__
                    if (fetchKey) {
                        // This slot's previous logical page precedes the current QK.
                        AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(7U + nextKey % RESIDENT_KEY_BANKS);
                    }
#endif
#ifdef __DAV_VEC__
                    const bool keyLane = AscendC::GetSubBlockIdx() == 0U;
                    if (keyLane ? fetchKey : nextMiss && firstHeadStep) {
                        PrefetchResidentKv(halfDequant, params, gBlockTable[blockBOffset], kvHeadIdx,
                                          keyLane ? nextKey : kvSplitStart + nextPage, keyLane);
                    }
#endif
                }
                const uint32_t kvSTileIdx = kvSplitStart + (paired ? kvSTileRelIdx / 2U : kvSTileRelIdx);
                const uint32_t pipelineTileSeq =
                    flashDecode ? pipelineTileBase + kvSTileRelIdx : kvSTileRelIdx;
#ifdef __DAV_VEC__
                if constexpr (PairedKv) {
                    const bool keyLane = AscendC::GetSubBlockIdx() == 0U;
                    const auto cache = keyLane ? params.k : params.v;
                    const auto scale = keyLane ? params.kDescale : params.vDescale;
                    const uint32_t l1Base = keyLane ? qkL1TileM_ * qkL1TileKLeft_ * sizeof(ElementQ) * qL1BufNum_ :
                        pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * sizeof(ElementP) * pL1BufNum_;
                    const uint32_t l1Pitch = keyLane ? qkL1TileKRight_ * qkL1TileN_ * sizeof(ElementK) :
                        pvL1TileKRight_ * pvL1TileN_ * sizeof(ElementV);
                    // K starts with both members, then stays one QK ahead. V follows Softmax
                    // one step later, before the corresponding PV consumes its L1 credit.
                    const uint32_t passes = keyLane && kvSTileRelIdx == 0U ? 2U : 1U;
                    for (uint32_t pass = 0; pass < passes; ++pass) {
                        if (!keyLane && kvSTileRelIdx == 0U) { continue; }
                        const uint32_t seq = keyLane ? (kvSTileRelIdx == 0U ? pass : kvSTileRelIdx + 1U)
                                                     : kvSTileRelIdx - 1U;
                        if (seq >= kvSLoopCount) { continue; }
                        const uint32_t member = seq % 2U;
                        const uint32_t addr = l1Base + member * l1Pitch;
                        const uint32_t flag = (keyLane ? 7U : 9U) + member;
                        fa950_mixed::PairedDecodeKv producer(resource, secondPlan.total);
                        if (kvSeqlen >= (kvHeads_ == 4U ? 16384 : 24576)) {
                            producer.Produce<true>(cache, scale, gBlockTable[blockBOffset], kvHeadIdx, kvHeads_,
                                kvSplitStart, pageCount, noSkipKvS, seq, addr, flag);
                        } else {
                            producer.Produce<false>(cache, scale, gBlockTable[blockBOffset], kvHeadIdx, kvHeads_,
                                kvSplitStart, pageCount, noSkipKvS, seq, addr, flag);
                        }
                    }
                } else if constexpr (PairedHeads && MixedKv) {
                    if (!ResidentPages && kvSTileRelIdx < kvSLoopCount) {
                        ProducePairedKv(halfDequant, params, gBlockTable[blockBOffset], kvHeadIdx, kvSplitStart,
                                        kvSplitEnd, kvSLoopNum, noSkipKvS, pvL1AddrStart_, kvSTileRelIdx);
                    }
                } else if (mixedKv) {
                    // Small-Q: prepare V(i) while Cube consumes V(i-2). The
                    // existing two L1 slots gate DMA reuse, not the VF itself.
                    // P(i-2) was published before any wait for that slot's reuse.
                    // With one staging output, produce V(i-1) before K(i).
                    // Its L1 slot was last consumed by PV(i-3) in the prior
                    // iteration, so reuse does not depend on the current QK.
                    const uint32_t vProductionDelay = (earlyKv || splitRowKv) ? 0U : PRE_LAUNCH - 1U;
                    if (kvSTileRelIdx >= vProductionDelay &&
                        kvSTileRelIdx - vProductionDelay < kvSLoopCount) {
                        const uint32_t rel = kvSTileRelIdx - vProductionDelay;
                        const uint32_t tile = kvSplitStart + rel;
                        const uint32_t seq = flashDecode ? pipelineTileBase + rel : rel;
                        const uint32_t rows = tile == kvSLoopNum - 1 ? noSkipKvS - tile * 128U : 128U;
                        const uint32_t offset = pvL1AddrStart_ + pvL1TileM_ * pvL1TileKLeft_ * 2U * pL1BufNum_
                            + (seq % vL1BufNum_) * pvL1TileKRight_ * pvL1TileN_ * 2U;
                        if (bufferedKv) {
                            // Keep one input slot free so its next DMA can precede this VF.
                            if (earlyKv && rel + kvPrefetchDistance < kvSLoopCount) {
                                const uint32_t nextTile = tile + kvPrefetchDistance;
                                const uint32_t nextRows = nextTile == kvSLoopNum - 1 ? noSkipKvS - nextTile * 128U : 128U;
                                bufferedDequant.Prefetch(params.v, params.vDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                                    params.storage.valueHeadStride, nextTile, nextRows, 0U);
                            }
                            bufferedDequant.Produce(resource, rows - (rel == 0U ? firstPageSkip : 0U), offset, 9U + seq % vL1BufNum_, 0U);
                            // Queue the next read before Softmax/Rescale. Input
                            // reuse waits only for this VF; output remains protected.
                            if (!earlyKv && rel + kvPrefetchDistance < kvSLoopCount) {
                                const uint32_t nextTile = tile + kvPrefetchDistance;
                                const uint32_t nextRows = nextTile == kvSLoopNum - 1 ? noSkipKvS - nextTile * 128U : 128U;
                                bufferedDequant.Prefetch(params.v, params.vDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                                    params.storage.valueHeadStride, nextTile, nextRows, 0U);
                            }
                        } else if (splitRowKv) {
                            halfDequant.Produce(resource, rows, offset, 9U + seq % vL1BufNum_, 0U);
                            if (rel + 1U < kvSLoopCount) {
                                const uint32_t next = tile + 1U;
                                const uint32_t nextRows = next == kvSLoopNum - 1 ? noSkipKvS - next * 128U : 128U;
                                halfDequant.Prefetch(params.v, params.vDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                                    params.storage.valueHeadStride, next, nextRows, 0U);
                            }
                        } else {
                            kvDequant.Produce(resource, params.v, params.vDescale, gBlockTable[blockBOffset],
                            kvHeadIdx, kvHeads_, params.storage.valuePageStride, params.storage.valueTokenStride,
                            params.storage.valueHeadStride, tile, rows, offset, 9U + seq % vL1BufNum_);
                        }
                    }
                    const uint32_t kRel = kvSTileRelIdx + ((earlyKv || splitRowKv) ? 1U : 0U);
                    const uint32_t kTile = kvSplitStart + kRel;
                    const uint32_t kSeq = flashDecode ? pipelineTileBase + kRel : kRel;
                    if (kRel < kvSLoopCount) {
                        const uint32_t rows = kTile == kvSLoopNum - 1 ? noSkipKvS - kTile * 128U : 128U;
                        const uint32_t offset = qkL1TileM_ * qkL1TileKLeft_ * 2U * qL1BufNum_
                            + (kSeq % kL1BufNum_) * qkL1TileKRight_ * qkL1TileN_ * 2U;
                        if (bufferedKv) {
                            // Keep one input slot free so its next DMA can precede this VF.
                            if (earlyKv && kRel + kvPrefetchDistance < kvSLoopCount) {
                                const uint32_t nextTile = kTile + kvPrefetchDistance;
                                const uint32_t nextRows = nextTile == kvSLoopNum - 1 ? noSkipKvS - nextTile * 128U : 128U;
                                bufferedDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                                    params.storage.keyHeadStride, nextTile, nextRows, 1U);
                            }
                            if (AscendC::GetSubBlockIdx() == 0U) {
                                bufferedDequant.Produce(resource, rows, offset, 7U + kSeq % kL1BufNum_, 1U);
                            }
                            // Queue the next read before Softmax/Rescale. Input
                            // reuse waits only for this VF; output remains protected.
                            if (!earlyKv && kRel + kvPrefetchDistance < kvSLoopCount) {
                                const uint32_t nextTile = kTile + kvPrefetchDistance;
                                const uint32_t nextRows = nextTile == kvSLoopNum - 1 ? noSkipKvS - nextTile * 128U : 128U;
                                bufferedDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                                    params.storage.keyHeadStride, nextTile, nextRows, 1U);
                            }
                        } else if (splitRowKv) {
                            halfDequant.Produce(resource, rows, offset, 7U + kSeq % kL1BufNum_, 1U);
                            if (kRel + 1U < kvSLoopCount) {
                                const uint32_t next = kTile + 1U;
                                const uint32_t nextRows = next == kvSLoopNum - 1 ? noSkipKvS - next * 128U : 128U;
                                halfDequant.Prefetch(params.k, params.kDescale, gBlockTable[blockBOffset],
                                    kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                                    params.storage.keyHeadStride, next, nextRows, 1U);
                            }
                        } else {
                            kvDequant.Produce(resource, params.k, params.kDescale, gBlockTable[blockBOffset],
                            kvHeadIdx, kvHeads_, params.storage.keyPageStride, params.storage.keyTokenStride,
                            params.storage.keyHeadStride, kTile, rows, offset, 7U + kSeq % kL1BufNum_);
                        }
                    }
                }
#endif
                if (kvSTileRelIdx < kvSLoopCount) {
                    if (kvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    if constexpr (SingleQueryWindow) {
                        if (kvSTileIdx == kvStart) { kvSTileSizeAct -= firstPageSkip; }
                    }
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = PairedHeads && !PairedKv ? 0U : pipelineTileSeq % UB_S_OTMP_BUF_STAGES;
                    int64_t stride = 64;
                    auto ubSLayoutTla =
                        tla::MakeLayout<ElementS, LayoutS>(qkRowNumRound, RoundUp(kvSTileSizeAct, ubSRoundTile));
                    auto ubSLayoutTlaDN = tla::MakeLayout<ElementS, LayoutS>(RoundUp(kvSTileSizeAct, 16), 64);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    auto ubSTensorTlaDN = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTlaDN, Arch::PositionUB{});
                    uint32_t qkReadyFlagId = ubSBufId;
                    Arch::CrossCoreFlag qkReadyFlag(qkReadyFlagId);
#ifdef __DAV_CUBE__
                    const uint64_t prefixSumL0AStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdx, qkL0ATotalStages_, pvL0ATotalStages_,
                                                         kvSLoopCount, true) +
                        (flashDecodeFlag_ != 0U ? pipelineL0AStageBase : 0U);
                    const uint64_t prefixSumL0BStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdx, qkL0BTotalStages_, pvL0BTotalStages_,
                                                         kvSLoopCount, true) +
                        (flashDecodeFlag_ != 0U ? pipelineL0BStageBase : 0U);
                    if constexpr (IsDN) {
                        blockMmadQK(gmKTensorTlaDN, ubSTensorTlaDN, gBlockTable[blockBOffset], actualBlockShapeQK,
                                    blockSize_, kvSTileIdx, kvSTileRelIdx, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0, 0,
                                    0, qSBlockSize, qNBlockSize, qkReadyFlag, prefixSumL0AStages, prefixSumL0BStages);
                    } else {
                        blockMmadQK.qBank = paired ? kvSTileRelIdx % 2U : 0U;
                        if (deferSecondQ && kvSTileRelIdx == 1U) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(leadKeyPages ? EVENT_ID0 : EVENT_ID5);
                        }
                        blockMmadQK.kPagePrefetched =
                            leadKeyPages && paired && (kvSTileRelIdx & 1U) == 0U &&
                            (kvSTileRelIdx >= 2U || reuseFirstPage);
                        blockMmadQK.template operator()<ResidentPages>(
                                    gmKTensorTla, ubSTensorTla, gBlockTable[blockBOffset], actualBlockShapeQK,
                                    blockSize_, kvSTileIdx, pipelineTileSeq, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0,
                                    0, 0, qSBlockSize, qNBlockSize, qkReadyFlag, prefixSumL0AStages,
                                    prefixSumL0BStages, ResidentPages ? kvSTileIdx % RESIDENT_KEY_BANKS : 0U,
                                    ResidentPages && kvSTileIdx >= firstKeyMiss);
                        if (deferSecondQ && kvSTileRelIdx == 0U) {
                            // Queue head 1 after the first K DMA, while the
                            // Cube and Vector pipelines process head 0.
                            auto nextQ = tla::MakeTensor(gQ[gmOffsetQ + qHeadStride], gmQLayoutTla, Arch::PositionGM{});
                            blockMmadQK.qBank = 1U;
                            blockMmadQK.loadQGM(nextQ, actualBlockShapeQ, qSBlockSize, qNBlockSize,
                                                qHeadStride, false, BlockMmadQK::QLoadMode::DeferredHead1,
                                                leadKeyPages ? EVENT_ID0 : EVENT_ID5);
                        }
                    }
                    if (prefetchNativeKv && paired && (kvSTileRelIdx & 1U) == 0U && kvSTileIdx + 1U < kvSplitEnd) {
                        // V(i+1) is one paired page ahead. The oldest occupant
                        // of its bank retired in the preceding iteration.
                        blockMmadPV.PrefetchNativeV(gmVTensorTla, gBlockTable[blockBOffset], kvSTileIdx + 1U, embedV_, noSkipKvS, blockSize_);
                    }
                    if (kvSTileRelIdx == kvSLoopCount - 1U) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    uint32_t l1PBufId = pipelineTileSeq % pL1BufNum_;
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, Catlass::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    // Work queued on the two L1 P banks still produces a
                    // three-slot ring of online statistics for rescale.
                    const uint32_t dmSlot = compactPipeline ?
                        pipelineTileSeq % MAX_CROSS_CORE_BUF_STAGES : l1PBufId;
                    auto& epilogueOnlineSoftmax = paired && kvSTileRelIdx % 2U ? secondSm : firstSm;
                    if constexpr (SingleQueryWindow) {
                        // The first and last KV pages have already been clipped to the single query window.
                        epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx == 0U, ubSBufId,
                                              dmSlot, qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize);
                    } else if constexpr (maskCategory == MaskCategory::MASK_CAUSAL || maskCategory == MaskCategory::MASK_SWA) {
                        const int64_t key0 = static_cast<int64_t>(kvSTileIdx) * kvBaseTile_;
                        const int64_t minDiag = 1 - static_cast<int64_t>(qSBlockSize);
                        const int64_t maxDiag = static_cast<int64_t>(kvSTileSizeAct) - 1;
                        const bool maskLeft = !notPreMask && windowSizeLeftEndLen - 1 > key0;
                        const bool maskRight = !notNextMask && windowSizeRightStartLen < key0 + maxDiag;
                        if (maskCategory == MaskCategory::MASK_SWA && PairedHeads && qNBlockSize == 1U &&
                            kvSTileSizeAct > 64U && maskLeft != maskRight) {
                            constexpr bool alignedSwaEdge = ResidentPages || nativePair;
                            // Lower bound is inclusive; the upper boundary counts valid columns.
                            const int64_t boundary = (maskLeft ? windowSizeLeftStartLen :
                                windowSizeRightStartLen + 1) - key0;
                            const int32_t edge = static_cast<int32_t>(AscendC::Std::min(int64_t{128},
                                AscendC::Std::max(int64_t{-128}, boundary)));
                            if (maskLeft) {
                                epilogueOnlineSoftmax.template operator()<false, false, Epilogue::Block::ScoreBoundary::Lower,
                                                                          MixedKv, alignedSwaEdge>(
                                    l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx < headCount, ubSBufId, dmSlot,
                                    qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize, edge);
                            } else {
                                epilogueOnlineSoftmax.template operator()<false, false, Epilogue::Block::ScoreBoundary::Upper,
                                                                          MixedKv, alignedSwaEdge>(
                                    l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx < headCount, ubSBufId, dmSlot,
                                    qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize, edge);
                            }
                        } else if (maskLeft || maskRight) {
                            // For this tile, (key column - query row within head) lies in [minDiag,maxDiag].
                            // Clip in signed 64-bit before feeding the VF, including fully masked tiles.
                            const int64_t lo = maskLeft ? AscendC::Std::min(windowSizeLeftStartLen - key0, maxDiag + 1) : minDiag;
                            const int64_t hi = maskRight ? AscendC::Std::max(windowSizeRightStartLen - key0, minDiag - 1) : maxDiag;
                            const int32_t lower = static_cast<int32_t>(lo <= hi ? lo : maxDiag + 1);
                            const uint32_t width = lo <= hi ? static_cast<uint32_t>(hi - lo) : 0U;
                            if (qNBlockSize == 1U) {
                                epilogueOnlineSoftmax.template operator()<true, false>(
                                    l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx < headCount, ubSBufId, dmSlot,
                                    qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize, lower, width);
                            } else {
                                epilogueOnlineSoftmax.template operator()<true, true>(
                                    l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx < headCount, ubSBufId, dmSlot,
                                    qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize, lower, width);
                            }
                        } else {
                            epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, kvSTileRelIdx < headCount, ubSBufId,
                                                  dmSlot, qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize);
                        }
                    } else {
                        if constexpr (IsDN) {
                            epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx < headCount), ubSBufId,
                                                  dmSlot, qkReadyFlag, softmaxReadyFlag, 1, qSBlockSize, qNBlockSize);
                        } else {
                            epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvSTileRelIdx < headCount), ubSBufId,
                                                  dmSlot, qkReadyFlag, softmaxReadyFlag, qSBlockSize, qNBlockSize);
                        }
                    }
#endif
                }
#ifdef __DAV_VEC__
                if constexpr (ResidentPages) {
                    const bool keyLane = AscendC::GetSubBlockIdx() == 0U;
                    if (keyLane && fetchKey) {
                        ProduceResidentKv(halfDequant, nextKey % RESIDENT_KEY_BANKS, pvL1AddrStart_, true);
                    } else if (!keyLane && nextMiss && !lateNext && !firstHeadStep) {
                        ProduceResidentKv(halfDequant, residentWindow.banks[nextPage], pvL1AddrStart_, false);
                    } else if (!keyLane && retireTailPredecessor) {
                        ProduceResidentKv(halfDequant, residentWindow.banks[pageCount - 1U], pvL1AddrStart_, false);
                    }
                }
#endif
                if (kvSTileRelIdx >= PRE_LAUNCH) {
                    const uint32_t kvSTileRelIdxNow = kvSTileRelIdx - PRE_LAUNCH;
                    const uint32_t kvSTileIdxNow = kvSplitStart + (paired ? kvSTileRelIdxNow / 2U : kvSTileRelIdxNow);
                    const uint32_t pipelineTileSeqNow =
                        flashDecode ? pipelineTileBase + kvSTileRelIdxNow : kvSTileRelIdxNow;
                    if (kvSTileIdxNow == kvSLoopNum - 1) {
                        kvSTileSizeAct = noSkipKvS - kvSTileIdxNow * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    if constexpr (SingleQueryWindow) {
                        if (kvSTileIdxNow == kvStart) { kvSTileSizeAct -= firstPageSkip; }
                    }
                    GemmCoord actualBlockShapePV{rowNum, embedV_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = pipelineTileSeqNow % otmpStages_;
                    const uint32_t pFlagCount = compactPipeline ? MAX_CROSS_CORE_BUF_STAGES : pL1BufNum_;
                    uint32_t pvReadyFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pFlagCount;
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = pipelineTileSeqNow % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedVRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t softmaxReadyFlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag softmaxReadyFlag(softmaxReadyFlagId);
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    const uint64_t prefixSumL0AStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdxNow, qkL0ATotalStages_, pvL0ATotalStages_,
                                                         kvSLoopCount, false) +
                        (flashDecodeFlag_ != 0U ? pipelineL0AStageBase : 0U);
                    const uint64_t prefixSumL0BStages =
                        CalcCrossqkpvPrefixSumL0ABStages(kvSTileRelIdxNow, qkL0BTotalStages_, pvL0BTotalStages_,
                                                         kvSLoopCount, false) +
                        (flashDecodeFlag_ != 0U ? pipelineL0BStageBase : 0U);
                    const uint32_t page = kvSTileIdxNow - kvSplitStart;
                    blockMmadPV.template operator()<ResidentPages>(
                        gmVTensorTla, ubOTmpTensorTla, gBlockTable[blockBOffset], actualBlockShapePV,
                        blockSize_, kvSTileIdxNow, pipelineTileSeqNow, 0, kvHeads_, kvNumTokens, kvBaseTile_, 0,
                        0, 0, softmaxReadyFlag, pvReadyFlag, prefixSumL0AStages, prefixSumL0BStages,
                        ResidentPages ? residentWindow.banks[page] : 0U,
                        ResidentPages && residentWindow.Miss(page));
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag pvReadyFlag(pvReadyFlagId);
                    uint32_t curTileMod = pipelineTileSeqNow % (PRE_LAUNCH + 1);
                    uint32_t fmRowsPerHead = fullyMaskedRowsPerHead;
                    int32_t swaDelStartRow = 0;
                    int32_t swaDelEndRow = 0;
                    uint32_t swaQSeqlen = 0;
                    uint32_t swaQSTileStart = 0;
                    if constexpr (maskCategory == MaskCategory::MASK_SWA) {
                        fmRowsPerHead = 0U;
                        swaDelStartRow = delStartRow;
                        swaDelEndRow = delEndRow;
                        swaQSeqlen = static_cast<uint32_t>(qSeqlen);
                        swaQSTileStart = static_cast<uint32_t>(qSTileStart);
                    }
                    const uint32_t outMember = paired ? kvSTileRelIdxNow % 2U : 0U;
                    auto& epilogueRescaleO = outMember != 0U ? secondO : firstO;
                    auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO + outMember * qNBlockSize * oHeadStride], gmOLayoutTla, Arch::PositionGM{});
                    epilogueRescaleO.template operator()<LseMode>(
                        gmOTensorTla, gLse[lseOffset + outMember * qNBlockSize * lseHeadStride], actualBlockShapePV, curTileMod, pipelineTileSeqNow,
                        (kvSTileRelIdxNow < headCount), (kvSTileRelIdxNow + headCount >= kvSLoopCount), pvReadyFlag, IsDN,
                        fmRowsPerHead, qSBlockSize, qNBlockSize, lseHeadStride, static_cast<uint32_t>(strideO),
                        static_cast<uint32_t>(oHeadStride), swaDelStartRow, swaDelEndRow, swaQSeqlen, swaQSTileStart,
                        gPartialO, gPartialLse, writePartial && !IsDN, partialId + outMember * splitsPerHead, fdRowCapacity_, fdLseSubStride_);
#endif
                }
#ifdef __DAV_CUBE__
                if constexpr (ResidentPages) {
                    if (nextMiss && !lateNext && !firstHeadStep) {
                        // V trails K by one page. Queue the MTE3 credit after
                        // the last PV of the old page has read the bank.
                        AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(9U + residentWindow.banks[nextPage] + 16U);
                    }
                    if (retireTailPredecessor) {
                        AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(9U + residentWindow.banks[pageCount - 1U] + 16U);
                    }
                }
                if constexpr (!IsDN && nativePair && kvcacheShape == PageShape::BnBsND) {
                    if (leadKeyPages && paired && ((kvSTileRelIdx == 0U && pageCount >= 2U) ||
                        ((kvSTileRelIdx & 1U) != 0U && kvSTileRelIdx + 3U < kvSLoopCount))) {
                        const uint32_t page = kvSplitStart +
                            (kvSTileRelIdx == 0U ? 1U : kvSTileRelIdx / 2U + 2U);
                        blockMmadQK.PrefetchNativeKPage(gmKTensorTla, gBlockTable[blockBOffset],
                            page, blockSize_, embed_, noSkipKvS);
                    }
                }
#endif
                if constexpr (ResidentPages) {
                    nextKey += static_cast<uint32_t>(fetchKey);
                }
            }
            if (flashDecode) {
                pipelineTileBase += kvSLoopCount;
                pipelineL0AStageBase += static_cast<uint64_t>(kvSLoopCount) * (qkL0ATotalStages_ + pvL0ATotalStages_);
                pipelineL0BStageBase += static_cast<uint64_t>(kvSLoopCount) * (qkL0BTotalStages_ + pvL0BTotalStages_);
            }
        }
#ifdef __DAV_VEC__
        if (mixedKv) {
            if (bufferedKv) { bufferedDequant.Drain(); }
            else if (splitRowKv) { halfDequant.Drain(); }
            else { kvDequant.Drain(kvHeads_); }
            if constexpr (!ResidentPages) {
                for (uint32_t flag = 7; flag < 11; ++flag) {
                    AscendC::CrossCoreWaitFlag<4, PIPE_MTE3>(flag);
                }
            }
        }
#endif
#ifdef __DAV_CUBE__
        if constexpr (!IsDN && nativePair) {
            if (leadKeyPages) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BlockMmadQK::THIRD_K_BANK_EVENT);
            }
        }
#endif
        ReleaseSyncFlags<4, 4, 4>(compactPipeline);
        if constexpr (!IsDN && (qFormat == Format::TND || qFormat == Format::BSND) &&
                      kvcacheType == CacheMode::pagedCache &&
                      maskCategory != MaskCategory::MASK_SWA) {
            if (flashDecodeFlag_ != 0U && fdCombineTaskNum_ != 0U) {
                // All launched AIC/AIV blocks join after their partial DMAs
                // complete. Reuse the main kernel's UB only after this barrier.
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
                FlashDecodeCombine950<ElementO, LseMode, 0, qFormat == Format::BSND> combine(resource);
                combine(params);
#endif
            }
        }
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline void InitSyncFlags(bool prefetchV)
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        if (prefetchV) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BlockMmadPV::THIRD_V_BANK_EVENT);
        }
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        if (!prefetchV) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        }
        // cross core sync
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_PV_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // mask
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<QK_SM_MODE, PIPE_V>(1);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(5);
            if (otmpStages_ == 2U) {
                AscendC::CrossCoreSetFlag<PV_RE_MODE, PIPE_V>(6);
            }
        }
#endif
    }

    template <uint32_t QK_SM_MODE, uint32_t PV_RE_MODE, uint32_t SM_PV_MODE>
    __aicore__ inline void ReleaseSyncFlags(bool prefetchV)
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        if (prefetchV) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BlockMmadPV::THIRD_V_BANK_EVENT);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        if (!prefetchV) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        }
        if constexpr (QK_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<QK_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (PV_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(21);
            if (otmpStages_ == 2U) {
                AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(6);
                AscendC::CrossCoreWaitFlag<PV_RE_MODE, PIPE_FIX>(22);
            }
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_PV_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_PV_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint64_t CalcCrossqkpvPrefixSumL0ABStages(uint32_t KvSTileIdx, uint32_t singleqkL0Stages,
                                                                uint32_t singlepvL0Stages, uint32_t kvSLoopNum,
                                                                bool isCurPhaseqk)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseqk) {
            prefixSumStages = (KvSTileIdx <= PRE_LAUNCH)
                                  ? KvSTileIdx * singleqkL0Stages
                                  : KvSTileIdx * singleqkL0Stages + (KvSTileIdx - PRE_LAUNCH) * singlepvL0Stages;
        } else {
            prefixSumStages = (KvSTileIdx + PRE_LAUNCH < kvSLoopNum)
                                  ? (KvSTileIdx + 1 + PRE_LAUNCH) * singleqkL0Stages + KvSTileIdx * singlepvL0Stages
                                  : kvSLoopNum * singleqkL0Stages + KvSTileIdx * singlepvL0Stages;
        }
        return prefixSumStages;
    }

  private:
    Arch::ForwardResource950 resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t embedV_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    float softcapValue_;
    uint32_t maxNumBlocksPerBatch_;
    uint32_t blockSize_;
    uint32_t numBlocks_;
    uint32_t flashDecodeFlag_;
    uint32_t fdCombineTaskNum_;
    uint32_t fdRowCapacity_;
    uint32_t fdLseSubStride_;

    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    uint32_t actSeqAval_;

    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;

    uint32_t qkL1TileM_;
    uint32_t qkL1TileN_;
    uint32_t qkL1TileKLeft_;
    uint32_t qkL1TileKRight_;
    uint32_t pvL1TileM_;
    uint32_t pvL1TileN_;
    uint32_t pvL1TileKLeft_;
    uint32_t pvL1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;

    int64_t windowSizeLeft_ = 0;
    int64_t windowSizeRight_ = 0;

    uint32_t qkL0ATotalStages_;
    uint32_t qkL0BTotalStages_;
    uint32_t pvL0ATotalStages_;
    uint32_t pvL0BTotalStages_;
};

template <bool PipelineColumns, class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap, bool MixedKv = false,
          bool PairedHeads = false, bool SingleQueryWindow = false, bool PairedKv = false, bool GroupedHeads = false,
          bool ResidentPages = false>
__aicore__ inline void RunFAInfer(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                            GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
                            GM_ADDR workspace, GM_ADDR tiling, float scaleValue, float softcapValue, uint32_t numBlocks,
                            uint32_t maxNumBlocksPerBatch, int64_t kvSeqlenAllocated, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // To cater to the existing TileCopy class, a dummy rowMajor is used
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    // mask
    using LayoutMask = layout::RowMajor;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<Q_BLK>, Int<128>, Int<128>>;
    // Split the 192-wide reduction and output into balanced Cube fragments.
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<PipelineColumns ? 96 : 128>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmaxT<HasSoftcap>;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = std::conditional_t<PipelineColumns, Gemm::Block::PairedPrefillPvTile, Shape<Int<128>, Int<128>, Int<128>>>;
    using DispatchPolicyPV =
        Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, false>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementV, TileCopyRescaleO, Arch::PositionL0C>;

    using Kernel = FAIKernel950<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat,
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, false, LseMode,
                                MixedKv, PairedHeads, SingleQueryWindow, PairedKv, GroupedHeads, ResidentPages>;
    FAIKernelParams params{q,
                           k,
                           v,
                           mask,
                           blockTables,
                           actualQseqlen,
                           actualKvseqlen,
                           seqUsedQ,
                           seqUsedKv,
                           o,
                           lse,
                           workspace,
                           tiling,
                           scaleValue,
                           softcapValue,
                           numBlocks,
                           maxNumBlocksPerBatch,
                           kvSeqlenAllocated,
                           storage};
    params.kDescale = kDescale;
    params.vDescale = vDescale;
    AscendC::TPipe pipe;
    Kernel faInfer(pipe);
    faInfer(params);
}

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap, bool MixedKv = false, bool PairedHeads = false>
CATLASS_GLOBAL void FAInfer(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                            GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
                            GM_ADDR workspace, GM_ADDR tiling, float scaleValue, float softcapValue, uint32_t numBlocks,
                            uint32_t maxNumBlocksPerBatch, int64_t kvSeqlenAllocated, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    RunFAInfer<false, InDtype, SMDtype, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout,
               LseMode, HasSoftcap, MixedKv, PairedHeads>(
        q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv, workspace,
        tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated, storage, kDescale, vDescale);
}

template <bool GroupedHeads, MaskCategory Mask>
__aicore__ inline bool CanUseNativePairedPrefill(GM_ADDR tiling, GM_ADDR actualQseqlen,
                                                GM_ADDR actualKvseqlen, uint64_t totalQTokens)
{
    const auto* plan = reinterpret_cast<__gm__ FAInferTilingData*>(tiling);
    const auto* cu = reinterpret_cast<__gm__ int32_t*>(actualQseqlen);
    const auto* kv = reinterpret_cast<__gm__ int32_t*>(actualKvseqlen);
    const uint32_t qTile = plan->qBaseTile;
    const bool canPair64 = Mask == MaskCategory::MASK_SWA && qTile == 64U &&
        (plan->numHeads / plan->kvHeads) % 2U == 0U;
    if ((qTile != 128U && !canPair64) || plan->kvBaseTile != 128U || plan->flashDecodeFlag != 0U) {
        return false;
    }
    uint32_t tasks = plan->totalTaskNum / 2U;
    if constexpr (GroupedHeads) {
        const uint32_t heads = plan->numHeads;
        const uint32_t kvHeads = plan->kvHeads;
        const uint32_t tasksPerQTile = CeilDiv(heads / kvHeads, 2U) * kvHeads;
        tasks = plan->totalTaskNum / heads * tasksPerQTile;
    }
    if (tasks < AscendC::GetBlockNum()) {
        return false;
    }
    // Native pairing requires full Q tiles with right-aligned KV.
    for (uint32_t i = 0; i < plan->batch; ++i) {
        const int32_t end = cu[i + 1U];
        const int32_t rows = end - cu[i];
        if (rows < 0 || rows % qTile != 0 || kv[i] < rows) {
            return false;
        }
        if (static_cast<uint64_t>(end) == totalQTokens) {
            break;
        }
    }
    return true;
}

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap,
          bool MixedKv = false, bool PairedHeads = false, bool GroupedHeads = false>
CATLASS_GLOBAL void FAInferPairedPrefill(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                            GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
                            GM_ADDR workspace, GM_ADDR tiling, float scaleValue, float softcapValue, uint32_t numBlocks,
                            uint32_t maxNumBlocksPerBatch, int64_t kvSeqlenAllocated, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    if constexpr (PairedHeads && !MixedKv) {
        if (!CanUseNativePairedPrefill<GroupedHeads, maskCategory>(tiling, actualQseqlen, actualKvseqlen, storage.totalQTokens)) {
            RunFAInfer<false, InDtype, SMDtype, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory,
                       cacheLayout, LseMode, HasSoftcap>(
                q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv,
                workspace, tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated,
                storage, kDescale, vDescale);
            return;
        }
    }
    RunFAInfer<true, InDtype, SMDtype, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout,
               LseMode, HasSoftcap, MixedKv, PairedHeads, false, false, GroupedHeads>(
        q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv, workspace,
        tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated, storage, kDescale, vDescale);
}

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap, bool MixedKv = false, bool PairedHeads = false>
CATLASS_GLOBAL void FAInferSingleQueryWindow(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
                            GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
                            GM_ADDR workspace, GM_ADDR tiling, float scaleValue, float softcapValue, uint32_t numBlocks,
                            uint32_t maxNumBlocksPerBatch, int64_t kvSeqlenAllocated, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    RunFAInfer<false, InDtype, SMDtype, qFormat, kvFormat, kvcacheType, kvcacheShape, maskCategory, cacheLayout,
               LseMode, HasSoftcap, MixedKv, PairedHeads, true>(
        q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv, workspace,
        tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated, storage, kDescale, vDescale);
}

template <class InDtype, class SMDtype, Format qFormat, Format kvFormat, CacheMode kvcacheType, PageShape kvcacheShape,
          MaskCategory maskCategory, CacheLayout cacheLayout, bool LseMode, bool HasSoftcap>
CATLASS_GLOBAL void FAInferDn(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
                              GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ,
                              GM_ADDR seqUsedKv, GM_ADDR workspace, GM_ADDR tiling, float scaleValue,
                              float softcapValue, uint32_t numBlocks, uint32_t maxNumBlocksPerBatch,
                              int64_t kvSeqlenAllocated, FwdStorageParams storage, GM_ADDR kDescale, GM_ADDR vDescale)
{
    using ArchTag = Arch::Ascend950;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    static_assert(std::is_same_v<SMDtype, float>);
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = float;
    using ElementMask = uint8_t;
    using LayoutK = layout::RowMajor;
    using LayoutQ = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutMask = layout::RowMajor;
    using L1TileShapeQK = Shape<Int<128>, Int<Q_BLK>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK =
        Gemm::MmadFlashAttentionQK<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementK, LayoutK, ElementQ, LayoutQ, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_N>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ,
                                                  ElementS, void, TileCopyQK>;
    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueFAOnlineSoftmaxT<HasSoftcap>;
    using TileCopySoftmax = Epilogue::Tile::TileCopySoftmax<ArchTag, ElementMask, ElementP, LayoutMask, LayoutPDummy>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType, TileCopySoftmax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV =
        Gemm::MmadFlashAttentionPV<ArchTag, (bool)kvcacheType, (bool)cacheLayout, (bool)kvcacheShape, true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueFARescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementV, TileCopyRescaleO, Arch::PositionL0C>;

    using Kernel = FAIKernel950<BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat,
                                kvcacheType, kvcacheShape, maskCategory, cacheLayout, true, LseMode>;
    FAIKernelParams params{q,
                           k,
                           v,
                           mask,
                           blockTables,
                           actualQseqlen,
                           actualKvseqlen,
                           seqUsedQ,
                           seqUsedKv,
                           o,
                           lse,
                           workspace,
                           tiling,
                           scaleValue,
                           softcapValue,
                           numBlocks,
                           maxNumBlocksPerBatch,
                           kvSeqlenAllocated,
                           storage};
    AscendC::TPipe pipe;
    Kernel faInfer(pipe);
    faInfer(params);
}
