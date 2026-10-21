/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FA_TOKEN_MAJOR_SCHEDULE_H
#define FA_TOKEN_MAJOR_SCHEDULE_H

#include "fa_split.h"

namespace fa_split {

// One cohort contains num_heads_k cores. Its cores scan the SAME token
// interval for adjacent KV heads. Cohorts cycle over batches, and each core
// keeps a complete online-softmax reduction (no split-KV workspace/combine).
// Only shape/length facts belong here: actual KV strides are checked per call
// in the device kernel, so metadata remains reusable across BBND and BNBD.
inline uint32_t TokenMajorCoreCount(const SplitContext& ctx)
{
    if (ctx.batch_size <= 0 || ctx.num_heads_k <= 1 || ctx.num_heads <= 0 ||
        ctx.num_heads % ctx.num_heads_k != 0 || ctx.num_splits > 1 ||
        ctx.blockDim == 0 || ctx.blockDim % ctx.num_heads_k != 0 ||
        ctx.seqlens_k_cpu == nullptr) {
        return 0;
    }
    const uint32_t cohorts = ctx.blockDim / ctx.num_heads_k;
    const uint32_t batches = static_cast<uint32_t>(ctx.batch_size);
    // Avoid trading locality for idle cores or a large final-wave imbalance.
    const uint64_t paddedBatches = ((batches + cohorts - 1) / cohorts) * uint64_t(cohorts);
    if (batches < cohorts || paddedBatches * 4 > uint64_t(batches) * 5) {
        return 0;
    }
    const int32_t qLen = ctx.seqlens_q_cpu ? ctx.seqlens_q_cpu[0] : ctx.seqlen_q;
    const int32_t kvLen = ctx.seqlens_k_cpu[0];
    if (qLen <= 0 || kvLen <= 0) {
        return 0;
    }
    for (int32_t b = 1; b < ctx.batch_size; ++b) {
        if ((ctx.seqlens_q_cpu && ctx.seqlens_q_cpu[b] != qLen) || ctx.seqlens_k_cpu[b] != kvLen) {
            return 0;
        }
    }
    return ctx.blockDim;
}

} // namespace fa_split
#endif
