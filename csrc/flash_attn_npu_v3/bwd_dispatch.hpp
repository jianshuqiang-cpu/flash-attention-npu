// Copyright (c) 2026, Minghua Shen.
//
// Backward (FAGGeneral) kernel dispatch, split across two translation units by
// dtype (bwd_dispatch_bf16.cpp / bwd_dispatch_fp16.cpp), each owning 32 of the
// 64 FAGGeneral instantiations so they compile in parallel. flash_api.cpp
// computes all host-side setup (tiling, workspace, layout) and hands the raw
// device pointers / scalars to launch_bwd(); it picks dtype + layout at runtime
// and calls the matching dtype-specific launcher.
//
// This header stays lightweight (no CATLASS / kernel includes) so flash_api.cpp
// can include it without dragging in the heavy kernel templates.

#pragma once

#include <cstdint>
#include "acl/acl.h"

struct BwdLaunchArgs {
    uint32_t blockDim;
    aclrtStream aclStream;
    uint64_t fftsAddr;
    bool is_bf16;
    bool has_attn_mask;
    bool deterministic;
    uint32_t qk_headdim_kernel;  // 64 / 128 / 192 / 256
    uint32_t kInputLayout;       // BSND (0) or TND (1), see kernel_common_fag.hpp
    uint8_t *dOutDevice;
    uint8_t *qDevice;
    uint8_t *kDevice;
    uint8_t *vDevice;
    uint8_t *outDevice;
    uint8_t *attenMaskDevice;    // may be nullptr when has_attn_mask is false
    uint8_t *softMaxLseDevice;
    uint8_t *cuSeqQlenDevice;    // may be nullptr in BSND mode
    uint8_t *cuSeqKvlenDevice;   // may be nullptr in BSND mode
    uint8_t *dqDevice;
    uint8_t *dkDevice;
    uint8_t *dvDevice;
    uint8_t *workspaceDevice;
    uint8_t *tilingDevice;
};

// Per-dtype launchers, defined in bwd_dispatch_bf16.cpp / bwd_dispatch_fp16.cpp.
// Each picks the layout (BSND/TND) and instantiates the 32 FAGGeneral variants
// for its dtype (mask x deterministic x headdim x layout).
void launch_bwd_bf16(const BwdLaunchArgs &args);
void launch_bwd_fp16(const BwdLaunchArgs &args);

// Runtime entry: select dtype, then the dtype-specific launcher selects layout.
inline void launch_bwd(const BwdLaunchArgs &args) {
    if (args.is_bf16) {
        launch_bwd_bf16(args);
    } else {
        launch_bwd_fp16(args);
    }
}
