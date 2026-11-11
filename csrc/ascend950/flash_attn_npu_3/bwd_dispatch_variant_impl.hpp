/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FA950_BWD_DISPATCH_VARIANT_IMPL_HPP
#define FA950_BWD_DISPATCH_VARIANT_IMPL_HPP

#include "autogen/bwd_dispatch_variants.hpp"

// fag_kernel.cpp (and the FAG headers it pulls in) assume these standard
// headers are already visible. In the original single-TU layout they were
// supplied transitively by fag_tiling.cpp / torch/extension.h in mha_bwd.cpp,
// which are no longer included ahead of fag_kernel.cpp here.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// fag_kernel.cpp provides the FlashAttentionV3Bwd950 template and
// FAGTiling950::Layout; each leaf TU includes it once and instantiates exactly
// one (mask, deterministic, softcap) combination.
#include "fag_kernel.cpp"

// The dispatch carries layouts as plain uint32_t; keep the mirror enum in sync
// with the kernel-side enum at compile time.
static_assert(static_cast<uint32_t>(FAGTiling950::Layout::BSND) == BSND, "BwdInputLayout::BSND out of sync");
static_assert(static_cast<uint32_t>(FAGTiling950::Layout::TND) == TND, "BwdInputLayout::TND out of sync");
static_assert(static_cast<uint32_t>(FAGTiling950::Layout::BNSD) == BNSD, "BwdInputLayout::BNSD out of sync");

// One FlashAttentionV3Bwd950 instantiation per leaf TU. The 15 launch arguments
// are identical across every instantiation; only the template parameters
// differ.
template <typename DType, uint32_t kInputLayout, bool kMask, bool kDtm, bool kSoftcap>
void bwd_dispatch_variant(const BwdLaunchArgs& a)
{
    FlashAttentionV3Bwd950<DType, static_cast<FAGTiling950::Layout>(kInputLayout), kMask, kDtm, kSoftcap>
        <<<a.blockDim, nullptr, a.aclStream>>>(
            a.dOutDevice, a.qDevice, a.kDevice, a.vDevice, a.outDevice, a.attenMaskDevice,
            a.softMaxLseDevice, a.cuSeqQlenDevice, a.cuSeqKvlenDevice, a.seqUsedQDevice,
            a.seqUsedKvDevice, a.dqDevice, a.dkDevice, a.dvDevice, a.workspaceDevice,
            a.tilingDevice);
}

#endif // FA950_BWD_DISPATCH_VARIANT_IMPL_HPP
