/**
 * FlashDecode enable decision for FA4.
 * Runs the ported SplitCore balancer and turns FD on only when one Q row's
 * KV blocks are assigned to more than one core (numOfFdHead > 0).
 */
#ifndef FA_FD_ENABLE_H
#define FA_FD_ENABLE_H

#include "fa_split_core.h"

#include <cstdint>

namespace fa_fd {

// Q rows are packed on an S1*G axis up to 512, matching the high-perf GQA
// TND mBase used by FIA SplitCore. KV blocks are 512 tokens, matching both
// that sInner and FA4's MAX_KV_STACK_LEN.
constexpr uint32_t kMBaseSize = 512;
constexpr uint32_t kS2BaseSize = 512;
constexpr uint32_t kGS1BaseSizeOfFd = 8;
constexpr uint32_t kVecCubeRatio = 2;

inline bool EnableFlashDecodeBySplitCore(uint32_t batch, uint32_t numHeads, uint32_t numHeadsK, uint32_t maxQSeqlen,
                                         uint32_t coreNum, const int32_t* seqlensQ, const int32_t* seqlensK)
{
    if (batch == 0U || numHeadsK == 0U || numHeads < numHeadsK || (numHeads % numHeadsK) != 0U || coreNum == 0U ||
        seqlensK == nullptr) {
        return false;
    }

    fa_split_core::BaseInfo base{};
    base.bSize = batch;
    base.n2Size = numHeadsK;
    base.gSize = numHeads / numHeadsK;
    base.s1Size = maxQSeqlen;
    base.isS1G = true;
    base.attenMaskFlag = false;
    base.isAccumSeqS2 = false;
    base.actualLenKvDims = batch;
    base.actualSeqS2Size.reserve(batch);

    uint32_t maxKv = 0U;
    for (uint32_t i = 0; i < batch; ++i) {
        int64_t kv = seqlensK[i];
        if (kv < 0) {
            kv = 0;
        }
        base.actualSeqS2Size.push_back(kv);
        if (kv > static_cast<int64_t>(maxKv)) {
            maxKv = static_cast<uint32_t>(kv);
        }
    }
    base.s2Size = maxKv;

    if (seqlensQ != nullptr) {
        base.isAccumSeqS1 = false;
        base.actualLenQDims = batch;
        base.actualSeqS1Size.reserve(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            int64_t q = seqlensQ[i];
            if (q < 0) {
                q = 0;
            }
            base.actualSeqS1Size.push_back(q);
        }
    }

    fa_split_core::SplitParam param{};
    param.mBaseSize = kMBaseSize;
    param.s2BaseSize = kS2BaseSize;
    param.gS1BaseSizeOfFd = kGS1BaseSizeOfFd;

    fa_split_core::SplitResult result{coreNum, kVecCubeRatio};
    fa_split_core::SplitCore(coreNum, base, param, result);
    return result.numOfFdHead > 0U;
}

} // namespace fa_fd

#endif // FA_FD_ENABLE_H
