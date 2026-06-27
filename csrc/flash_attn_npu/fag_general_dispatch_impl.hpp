// Copyright (c) 2026, Minghua Shen.
//
// Shared implementation of the v2 FAGGeneral backward dispatch. Included by
// fag_general_dispatch_bf16.cpp and fag_general_dispatch_fp16.cpp, each of
// which explicitly instantiates launch_fag_general_dispatch_{bf16,fp16}<TND>
// and <BSND>, so the 64 FAGGeneral instantiations land in two parallel-compiled
// object files (32 each) instead of one.
//
// The launch tree reproduces the exact causal x deterministic x headdim
// combinations of LaunchFAGGeneralKernel in fag_general_launch.hpp; the dtype
// dimension is hoisted to a template parameter so each TU instantiates one
// dtype. Template params of ::FAGGeneral map as:
//   <DTemplateType::AlignedNNN, DType, kInputLayout, IS_CAUSAL, 0, IS_DTM>
// where IS_CAUSAL => IS_ATTEN_MASK, IS_DTM => deterministic (IS_DTM).

#pragma once

#include "fag_general_dispatch.hpp"

// fag_kernel.cpp (and the CATLASS/FAG headers it pulls in, e.g.
// kernel_common_fag.hpp) assume these standard headers are already visible.
// In the original single-TU layout they were supplied transitively by
// fag_tiling.cpp, which is NOT included here (its FAGTiling::* function
// definitions are host-side tiling helpers owned by fag_general_host.cpp;
// pulling them in would create duplicate symbols across the per-dtype TUs).
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

// fag_kernel.cpp provides the ::FAGGeneral kernel template, the DTemplateType
// enum, and the unqualified BSND/TND layout constants (0/1, from
// kernel_common_fag.hpp which it includes). It self-includes fag_tiling.h for
// the FAGTilingData type. Each dtype TU includes it once.
#include "../flash_attn_npu_v3/fag_kernel.cpp"

// Launch one FAGGeneral specialization. IS_CAUSAL / IS_DTM map to the kernel's
// IS_ATTEN_MASK / IS_DTM template params; ALN selects DTemplateType::AlignedNNN.
// Argument order is identical to LaunchFAGGeneralKernel in fag_general_launch.hpp.
#define GEN_LAUNCH(ALN, IS_CAUSAL, IS_DTM)                                       \
    ::FAGGeneral<DTemplateType::ALN, DType, kInputLayout, IS_CAUSAL, 0, IS_DTM>  \
        <<<a.blockDim, nullptr, a.aclStream>>>(                                  \
            a.fftsAddr, a.dOutDevice, a.qDevice, a.kDevice, a.vDevice,           \
            a.outDevice, nullptr, a.attenMaskDevice, a.softMaxLseDevice,         \
            a.cuSeqQlenDevice, a.cuSeqKvlenDevice, a.dqDevice, a.dkDevice,       \
            a.dvDevice, nullptr, a.workspaceDevice, a.tilingDevice)

template <typename DType, uint32_t kInputLayout>
void launch_fag_general_dispatch_impl(const FagGeneralLaunchArgs &a) {
    const uint32_t hd = a.qk_headdim_kernel;
    if (a.is_causal) {
        if (a.deterministic) {
            switch (hd) {
                case 64:  GEN_LAUNCH(Aligned64,  1, 1); break;
                case 128: GEN_LAUNCH(Aligned128, 1, 1); break;
                case 192: GEN_LAUNCH(Aligned192, 1, 1); break;
                case 256: GEN_LAUNCH(Aligned256, 1, 1); break;
                default: break;
            }
        } else {
            switch (hd) {
                case 64:  GEN_LAUNCH(Aligned64,  1, 0); break;
                case 128: GEN_LAUNCH(Aligned128, 1, 0); break;
                case 192: GEN_LAUNCH(Aligned192, 1, 0); break;
                case 256: GEN_LAUNCH(Aligned256, 1, 0); break;
                default: break;
            }
        }
    } else {
        if (a.deterministic) {
            switch (hd) {
                case 64:  GEN_LAUNCH(Aligned64,  0, 1); break;
                case 128: GEN_LAUNCH(Aligned128, 0, 1); break;
                case 192: GEN_LAUNCH(Aligned192, 0, 1); break;
                case 256: GEN_LAUNCH(Aligned256, 0, 1); break;
                default: break;
            }
        } else {
            switch (hd) {
                case 64:  GEN_LAUNCH(Aligned64,  0, 0); break;
                case 128: GEN_LAUNCH(Aligned128, 0, 0); break;
                case 192: GEN_LAUNCH(Aligned192, 0, 0); break;
                case 256: GEN_LAUNCH(Aligned256, 0, 0); break;
                default: break;
            }
        }
    }
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_bf16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<bfloat16_t, kInputLayout>(a);
}

template <uint32_t kInputLayout>
void launch_fag_general_dispatch_fp16(const FagGeneralLaunchArgs &a) {
    launch_fag_general_dispatch_impl<half, kInputLayout>(a);
}
