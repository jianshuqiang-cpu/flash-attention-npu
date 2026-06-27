// Copyright (c) 2026, Minghua Shen.
//
// Forward FAInfer kernel dispatch. This translation unit owns all 20
// SplitFuse::FAInfer<...> instantiations; flash_api.cpp no longer includes
// mha_fwd_kvcache.cpp directly, so the two compile independently.
//
// The dispatch is semantically identical to the original if/else launch tree
// that lived in flash_api.cpp::mha_fwd: the same dtype x paged x flash-decode x
// mask x layout combinations, the same runtime arguments, in the same order.

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

// mha_fwd_kvcache.cpp provides the SplitFuse::FAInfer kernel template, the
// FAInferKernel class, FAIKernelParams, and the FaiKenel enum namespace
// (from kernel_common.hpp).
#include "mha_fwd_kvcache.cpp"

// All 13 FAInfer launch arguments are identical across every instantiation;
// only the template parameters differ. Collapse the boilerplate into a macro so
// the 20 call sites cannot drift from each other. The 5th template param (IS_FD)
// and PagedCacheFlag are bool constants; MASK/LAYOUT are FaiKenel enum members.
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
template <typename DType>
static void launch_fwd_dtype(const FwdLaunchArgs &a) {
    const uint32_t launchBlockDim = a.launchBlockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool paged_KV = a.paged_KV;
    const bool is_causal = a.is_causal;
    const bool is_varlen_q = a.is_varlen_q;
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
            if (is_varlen_q) {
                if (flashDecodeFlag) {
                    FWD_LAUNCH(DType, true, true, MASK_CAUSAL, TND);
                } else {
                    FWD_LAUNCH(DType, true, false, MASK_CAUSAL, TND);
                }
            } else {
                FWD_LAUNCH(DType, true, false, MASK_CAUSAL, BSND);
            }
        } else {
            if (is_varlen_q) {
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
            if (is_varlen_q) {
                FWD_LAUNCH(DType, false, false, MASK_CAUSAL, TND);
            } else {
                FWD_LAUNCH(DType, false, false, MASK_CAUSAL, BSND);
            }
        } else {
            if (is_varlen_q) {
                FWD_LAUNCH(DType, false, false, NO_MASK, TND);
            } else {
                FWD_LAUNCH(DType, false, false, NO_MASK, BSND);
            }
        }
    }
}

#undef FWD_LAUNCH

void launch_fwd(const FwdLaunchArgs &args) {
    if (args.is_bf16) {
        launch_fwd_dtype<bfloat16_t>(args);
    } else {
        launch_fwd_dtype<half>(args);
    }
}
