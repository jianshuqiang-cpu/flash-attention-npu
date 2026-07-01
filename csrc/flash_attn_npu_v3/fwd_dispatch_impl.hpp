/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include "fwd_dispatch.hpp"

// Standard headers that the CATLASS/FAG headers (reached via mha_fwd_kvcache.cpp)
// assume to be already visible. In the original single-TU layout these were
// supplied transitively; supply them explicitly here.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "mha_fwd_kvcache.cpp"

#define FWD_LAUNCH(DTYPE, PAGED, IS_FD, MASK, LAYOUT)                                 \
    SplitFuse::FAInfer<DTYPE, DTYPE, float, PAGED, IS_FD,                             \
                       FaiKenel::MaskType::MASK, FaiKenel::inputLayout::LAYOUT,       \
                       Catlass::Epilogue::LseModeT::OUT_ONLY>                         \
        <<<launchBlockDim, nullptr, aclStream>>>(                                     \
            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice,        \
            oDevice, softmaxLseDevice, qSeqDevice, kvSeqDevice,                       \
            workspaceDevice, tilingDevice)

// Forward dispatch is symmetric across dtype: factor the paged/mask/layout tree
// into a dtype-templated helper so the bf16 and half branches stay in lockstep.
// IS_TND selects the input layout (false => BSND, true => TND) at compile time;
// each (dtype, IS_TND) pair is explicitly instantiated in its own autogen TU.
template <typename DType, bool IS_TND>
void launch_fwd_dtype(const FwdLaunchArgs &a) {
    const uint32_t launchBlockDim = a.launchBlockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool flashDecodeFlag = a.flashDecodeFlag;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
    uint8_t *maskDevice = a.maskDevice;
    uint8_t *blockTableDevice = a.blockTableDevice;
    uint8_t *oDevice = a.oDevice;
    uint8_t *softmaxLseDevice = a.softmaxLseDevice;
    uint8_t *qSeqDevice = a.qSeqDevice;
    uint8_t *kvSeqDevice = a.kvSeqDevice;
    uint8_t *workspaceDevice = a.workspaceDevice;
    uint8_t *tilingDevice = a.tilingDevice;

    if (paged_KV) {
        if (is_causal) {
            if constexpr (IS_TND) {
                if (flashDecodeFlag) {
                    FWD_LAUNCH(DType, true, true, MASK_CAUSAL, TND);
                } else {
                    FWD_LAUNCH(DType, true, false, MASK_CAUSAL, TND);
                }
            } else {
                FWD_LAUNCH(DType, true, false, MASK_CAUSAL, BSND);
            }
        } else {
            if constexpr (IS_TND) {
                if (flashDecodeFlag) {
                    FWD_LAUNCH(DType, true, true, NO_MASK, TND);
                } else {
                    FWD_LAUNCH(DType, true, false, NO_MASK, TND);
                }
            } else {
                FWD_LAUNCH(DType, true, false, NO_MASK, BSND);
            }
        }
    } else {
        if (is_causal) {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, false, false, MASK_CAUSAL, TND);
            } else {
                FWD_LAUNCH(DType, false, false, MASK_CAUSAL, BSND);
            }
        } else {
            if constexpr (IS_TND) {
                FWD_LAUNCH(DType, false, false, NO_MASK, TND);
            } else {
                FWD_LAUNCH(DType, false, false, NO_MASK, BSND);
            }
        }
    }
}

#undef FWD_LAUNCH
