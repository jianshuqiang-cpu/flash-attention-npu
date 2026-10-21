/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FA_TOKEN_MAJOR_SCHEDULE_H
#define FA_TOKEN_MAJOR_SCHEDULE_H

#include "fa_split.h"

namespace fa_split {

inline uint32_t TokenMajorCoreCountUniform(int32_t batchSize, int32_t numHeads, int32_t numHeadsK,
                                           int32_t qLen, int32_t kvLen, uint32_t blockDim,
                                           int32_t numSplits, int32_t kvNewSeqlen)
{
    if (batchSize <= 0 || numHeadsK <= 1 || numHeads <= 0 || numHeads % numHeadsK != 0 ||
        numSplits > 1 || kvNewSeqlen != 0 || blockDim == 0 || blockDim % numHeadsK != 0) {
        return 0;
    }
    const uint32_t cohorts = blockDim / static_cast<uint32_t>(numHeadsK);
    const uint32_t batches = static_cast<uint32_t>(batchSize);
    const uint64_t paddedBatches = ((batches + cohorts - 1) / cohorts) * uint64_t(cohorts);
    if (batches < cohorts || paddedBatches * 4 > uint64_t(batches) * 5) {
        return 0;
    }
    if (qLen <= 0 || kvLen <= 0) {
        return 0;
    }
    return blockDim;
}

// One cohort contains num_heads_k cores. Adjacent cores process adjacent KV
// heads of the same batch while every core scans the full S2 range. This keeps
// token-major BBND/BSND/TND reads coalesced and avoids split-KV workspace work.
inline uint32_t TokenMajorCoreCount(const SplitContext& ctx, bool kvSeqlensAreCumulative = false)
{
    if (ctx.seqlens_k_cpu == nullptr || ctx.batch_size <= 0) {
        return 0;
    }
    const int32_t qLen = ctx.is_varlen_q
        ? ctx.cu_seqlen_q_cpu[1] - ctx.cu_seqlen_q_cpu[0]
        : ctx.seqlen_q;
    const int32_t kvLen = kvSeqlensAreCumulative
        ? ctx.seqlens_k_cpu[1] - ctx.seqlens_k_cpu[0]
        : ctx.seqlens_k_cpu[0];
    if (TokenMajorCoreCountUniform(ctx.batch_size, ctx.num_heads, ctx.num_heads_k, qLen, kvLen,
                                   ctx.blockDim, ctx.num_splits, ctx.kvNewSeqlen) == 0) {
        return 0;
    }
    for (int32_t b = 1; b < ctx.batch_size; ++b) {
        const int32_t curQ = ctx.is_varlen_q
            ? ctx.cu_seqlen_q_cpu[b + 1] - ctx.cu_seqlen_q_cpu[b]
            : ctx.seqlen_q;
        const int32_t curKv = kvSeqlensAreCumulative
            ? ctx.seqlens_k_cpu[b + 1] - ctx.seqlens_k_cpu[b]
            : ctx.seqlens_k_cpu[b];
        if (curQ != qLen || curKv != kvLen) {
            return 0;
        }
    }
    return ctx.blockDim;
}

} // namespace fa_split
#endif
