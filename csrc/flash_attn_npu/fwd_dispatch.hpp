// Copyright (c) 2026, Minghua Shen.
//
// Forward (SplitFuse::FAInfer) kernel dispatch for v2, isolated from flash_api.cpp
// into per-dtype translation units (fwd_dispatch_bf16.cpp / fwd_dispatch_fp16.cpp)
// so the 20 FAInfer instantiations compile in parallel across cores.
//
// flash_api.cpp computes all host-side setup (tiling, workspace, layout) and hands
// the raw device pointers / scalars to launch_fwd<IS_TND>(); the dispatch selects
// dtype / paged / causal / flash-decode and launches the matching FAInfer<...>.
// IS_TND selects the input layout: false => BSND (mha_fwd_kvcache, mha_fwd),
// true => TND (mha_varlen_fwd). The flash-decode 8th template parameter is only
// used by the BSND path (mha_fwd_kvcache); it is compiled out for TND via
// `if constexpr`, so no FD+TND instantiation (which the original never uses) is
// introduced.
//
// This header stays lightweight (no CATLASS / kernel includes) so flash_api.cpp
// can include it without dragging in the heavy kernel templates.

#pragma once

#include <cstdint>
#include "acl/acl.h"

struct FwdLaunchArgs {
    uint32_t blockDim;
    aclrtStream aclStream;
    uint64_t fftsAddr;
    bool is_bf16;
    bool paged_KV;
    bool is_causal;
    bool flashDecodeFlag;       // only meaningful for the BSND (kvcache) path
    uint8_t *qDevice;
    uint8_t *kDevice;
    uint8_t *vDevice;
    uint8_t *maskDevice;        // may be nullptr when is_causal is false
    uint8_t *blockTableDevice;  // may be nullptr when paged_KV is false
    uint8_t *oDevice;
    uint8_t *softmaxLseDevice;
    uint8_t *qSeqDevice;
    uint8_t *kvSeqDevice;
    uint8_t *workspaceDevice;
    uint8_t *tilingDevice;
};

// Per-dtype implementation, defined in fwd_dispatch_bf16.cpp / fwd_dispatch_fp16.cpp.
// Each instantiates only its dtype's FAInfer variants (BSND: 6, TND: 4 => 10 each).
template <typename DType, bool IS_TND>
void launch_fwd_impl(const FwdLaunchArgs &a);

// Runtime entry: pick dtype, dispatch to the matching dtype TU. IS_TND is chosen
// at the call site (kvcache/mha_fwd => false, varlen_fwd => true).
template <bool IS_TND>
inline void launch_fwd(const FwdLaunchArgs &a) {
    if (a.is_bf16) {
        launch_fwd_impl<bfloat16_t, IS_TND>(a);
    } else {
        launch_fwd_impl<half, IS_TND>(a);
    }
}
