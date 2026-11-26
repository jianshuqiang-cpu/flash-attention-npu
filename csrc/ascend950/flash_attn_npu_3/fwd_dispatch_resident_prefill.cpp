// Copyright (c) 2026, Minghua Shen.

#include "fwd_dispatch.hpp"
#include "fwd_kernel.cpp"

namespace {

// Check actual device lengths once per worker, including during graph replay.
// Evenly sized groups of full tiles let every worker advance by one Q tile
// inside its lane, without changing the number of tasks recorded in metadata.
__aicore__ inline bool CanUseResidentPrefill(GM_ADDR tiling, GM_ADDR actualQseqlen,
    GM_ADDR usedKv, uint64_t totalQTokens)
{
    const auto* plan = reinterpret_cast<__gm__ const FAInferTilingData*>(tiling);
    if (plan->qBaseTile != 128U || plan->kvBaseTile != 128U || plan->blockSize != 128U ||
        plan->qL1BufNum != 2U || plan->pL1BufNum != 3U || plan->flashDecodeFlag != 0U ||
        plan->embeddingSize != 192U || plan->embeddingSizeV != 192U ||
        plan->windowSizeLeft != 511 || plan->windowSizeRight != 0 ||
        plan->numHeads < 2U || plan->kvHeads == 0U || usedKv == nullptr) {
        return false;
    }
    const uint32_t pairs = plan->numHeads / 2U;
    const uint32_t cores = AscendC::GetBlockNum();
    if (plan->numHeads % plan->kvHeads != 0U || (plan->numHeads / plan->kvHeads) % 2U != 0U ||
        cores == 0U || cores % pairs != 0U) {
        return false;
    }
    const uint32_t rowsPerWave = 128U * (cores / pairs);
    const auto* cu = reinterpret_cast<__gm__ const int32_t*>(actualQseqlen);
    const auto* kv = reinterpret_cast<__gm__ const int32_t*>(usedKv);
    int32_t begin = 0;
    for (uint32_t batch = 0; batch < plan->batch; ++batch) {
        const int32_t end = cu[batch + 1U];
        if (end < begin || static_cast<uint64_t>(end) > totalQTokens) {
            return false;
        }
        const uint32_t rows = static_cast<uint32_t>(end - begin);
        if (rows != 0U) {
            const int32_t kvRows = kv[batch];
            if (rows % rowsPerWave != 0U || kvRows < 0 ||
                static_cast<uint32_t>(kvRows) < rows || kvRows % 128 != 0) {
                return false;
            }
        }
        if (static_cast<uint64_t>(end) == totalQTokens) {
            return totalQTokens / 128U * plan->numHeads == plan->totalTaskNum;
        }
        begin = end;
    }
    return false;
}

} // namespace

template <bool LseMode>
CATLASS_GLOBAL void FAInferResidentPrefill(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR lse, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR seqUsedQ, GM_ADDR seqUsedKv,
    GM_ADDR workspace, GM_ADDR tiling, float scaleValue, float softcapValue, uint32_t numBlocks,
    uint32_t maxNumBlocksPerBatch, int64_t kvSeqlenAllocated, FwdStorageParams storage,
    GM_ADDR kDescale, GM_ADDR vDescale)
{
    if (CanUseResidentPrefill(tiling, actualQseqlen, seqUsedKv, storage.totalQTokens)) {
        RunFAInfer<true, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND,
                   MaskCategory::MASK_SWA, CacheLayout::nd, LseMode, false, true, true, false, false, false, true>(
            q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv, workspace,
            tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated, storage, kDescale, vDescale);
    } else {
        RunFAInfer<true, bfloat16_t, float, Format::TND, Format::TND, CacheMode::pagedCache, PageShape::BnBsND,
                   MaskCategory::MASK_SWA, CacheLayout::nd, LseMode, false, true, true>(
            q, k, v, mask, blockTables, o, lse, actualQseqlen, actualKvseqlen, seqUsedQ, seqUsedKv, workspace,
            tiling, scaleValue, softcapValue, numBlocks, maxNumBlocksPerBatch, kvSeqlenAllocated, storage, kDescale, vDescale);
    }
}

template <bool LseMode>
static void LaunchResidentPrefill(const FwdLaunchArgs& a)
{
    FAInferResidentPrefill<LseMode><<<a.block_dim, nullptr, a.stream>>>(
        a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device, a.o_device, a.lse_device,
        a.q_seq_device, a.kv_seq_device, a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,
        a.tiling_device, a.scale_value, a.softcap_value, a.num_blocks, a.max_num_blocks_per_batch,
        a.kv_seqlen_allocated, a.storage, a.k_descale_device, a.v_descale_device);
}

void launch_resident_prefill(const FwdLaunchArgs& a)
{
    if (a.lse_mode) {
        LaunchResidentPrefill<true>(a);
    } else {
        LaunchResidentPrefill<false>(a);
    }
}
