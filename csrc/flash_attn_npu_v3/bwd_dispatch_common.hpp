// Copyright (c) 2026, Minghua Shen.
//
// Shared declarations/macros for the backward FAGGeneral dispatch, split across
// two translation units (bwd_dispatch_bf16.cpp / bwd_dispatch_fp16.cpp) so each
// dtype's 32 kernel instantiations compile in parallel. Keeping the launch
// boilerplate in this header means the two dtype files cannot drift from each
// other.

#pragma once

#include "bwd_dispatch.hpp"

// fag_kernel.cpp (and the CATLASS/FAG headers it pulls in, e.g.
// kernel_common_fag.hpp) assume these standard headers are already visible.
// In the original single-TU layout they were supplied transitively by
// fag_tiling.cpp, which is no longer included ahead of fag_kernel.cpp here.
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

// fag_kernel.cpp provides the FAGGeneral template, the DTemplateType enum,
// the FlashAttentionScoreGrad kernel class and the unqualified BSND/TND layout
// constants (0/1, from kernel_common_fag.hpp). Each dtype TU includes it once.
#include "fag_kernel.cpp"

// All 17 FAGGeneral launch arguments are identical across every instantiation;
// only the template parameters differ. Collapse the boilerplate into a macro so
// the call sites cannot drift from each other.
#define BWD_LAUNCH(DT, DTYPE, IS_MASK, IS_DTM)                                       \
    do {                                                                             \
        FAGGeneral<DT, DTYPE, kInputLayout, IS_MASK, 0, IS_DTM>                      \
            <<<blockDim, nullptr, aclStream>>>(                                      \
                fftsAddr, dOutDevice, qDevice, kDevice, vDevice, outDevice,          \
                nullptr, attenMaskDevice, softMaxLseDevice,                          \
                cuSeqQlenDevice, cuSeqKvlenDevice,                                   \
                dqDevice, dkDevice, dvDevice, nullptr, workspaceDevice, tilingDevice); \
    } while (0)

// Pick the headdim specialization at runtime.
#define BWD_LAUNCH_HD(DTYPE, IS_MASK, IS_DTM)              \
    do {                                                    \
        switch (qk_headdim_kernel) {                        \
            case 64:                                        \
                BWD_LAUNCH(DTemplateType::Aligned64, DTYPE, IS_MASK, IS_DTM);  \
                break;                                      \
            case 128:                                       \
                BWD_LAUNCH(DTemplateType::Aligned128, DTYPE, IS_MASK, IS_DTM); \
                break;                                      \
            case 192:                                       \
                BWD_LAUNCH(DTemplateType::Aligned192, DTYPE, IS_MASK, IS_DTM); \
                break;                                      \
            case 256:                                       \
                BWD_LAUNCH(DTemplateType::Aligned256, DTYPE, IS_MASK, IS_DTM); \
                break;                                      \
            default:                                        \
                break;                                      \
        }                                                   \
    } while (0)

// Bind the BwdLaunchArgs fields to the names the launch macros expect, then run
// the dtype-specific mask x deterministic x headdim selection. This inline
// template is instantiated (per dtype + layout) only in the TU that calls it, so
// each dtype's FAGGeneral variants land in their own .o.
template <typename DType, uint32_t kInputLayout>
inline void bwd_dispatch_run(const BwdLaunchArgs &a) {
    const uint32_t blockDim = a.blockDim;
    const aclrtStream aclStream = a.aclStream;
    const uint64_t fftsAddr = a.fftsAddr;
    const bool has_attn_mask = a.has_attn_mask;
    const bool deterministic = a.deterministic;
    const uint32_t qk_headdim_kernel = a.qk_headdim_kernel;
    uint8_t *dOutDevice = a.dOutDevice;
    uint8_t *qDevice = a.qDevice;
    uint8_t *kDevice = a.kDevice;
    uint8_t *vDevice = a.vDevice;
    uint8_t *outDevice = a.outDevice;
    uint8_t *attenMaskDevice = a.attenMaskDevice;
    uint8_t *softMaxLseDevice = a.softMaxLseDevice;
    uint8_t *cuSeqQlenDevice = a.cuSeqQlenDevice;
    uint8_t *cuSeqKvlenDevice = a.cuSeqKvlenDevice;
    uint8_t *dqDevice = a.dqDevice;
    uint8_t *dkDevice = a.dkDevice;
    uint8_t *dvDevice = a.dvDevice;
    uint8_t *workspaceDevice = a.workspaceDevice;
    uint8_t *tilingDevice = a.tilingDevice;
    (void)blockDim; (void)aclStream; (void)fftsAddr; (void)has_attn_mask;
    (void)deterministic; (void)qk_headdim_kernel;

    if (has_attn_mask) {
        if (deterministic) {
            BWD_LAUNCH_HD(DType, 1, 1);
        } else {
            BWD_LAUNCH_HD(DType, 1, 0);
        }
    } else {
        if (deterministic) {
            BWD_LAUNCH_HD(DType, 0, 1);
        } else {
            BWD_LAUNCH_HD(DType, 0, 0);
        }
    }
}

#undef BWD_LAUNCH_HD
#undef BWD_LAUNCH
