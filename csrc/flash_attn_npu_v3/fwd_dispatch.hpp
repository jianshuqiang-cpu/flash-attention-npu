// Copyright (c) 2026, Minghua Shen.
//
// Forward (FAInfer) kernel dispatch, isolated into its own translation unit so
// the 20 SplitFuse::FAInfer template instantiations (2 dtype x {paged/non-paged
// with flash-decode variants} x {causal/no-mask} x {TND/BSND}) are compiled
// separately from flash_api.cpp. flash_api.cpp computes all host-side setup
// (tiling, workspace, layout, block-table/mask pointers) and hands the raw
// device pointers / scalars to launch_fwd().

#pragma once

#include <cstdint>
#include "acl/acl.h"

struct FwdLaunchArgs {
    uint32_t launchBlockDim;
    aclrtStream aclStream;
    uint64_t fftsAddr;
    bool is_bf16;
    bool paged_KV;
    bool is_causal;
    bool is_varlen_q;
    bool flashDecodeFlag;
    uint8_t *qDevice;
    uint8_t *kDevice;
    uint8_t *vDevice;
    uint8_t *maskDevice;          // may be nullptr when is_causal is false
    uint8_t *blockTableDevice;    // may be nullptr when paged_KV is false
    uint8_t *oDevice;
    uint8_t *softmaxLseDevice;
    uint8_t *qSeqDevice;
    uint8_t *kvSeqDevice;
    uint8_t *workspaceDevice;
    uint8_t *tilingDevice;
};

// Launch the FAInfer forward kernel. Selects dtype / paged / flash-decode /
// mask / layout at runtime and instantiates the corresponding kernel template.
void launch_fwd(const FwdLaunchArgs &args);
