// Copyright (c) 2026, Minghua Shen.
//
// fp16 half of the backward FAGGeneral dispatch: the 32 half instantiations
// (mask x deterministic x headdim x BSND/TND). Compiled in its own translation
// unit so it runs in parallel with the bf16 half and the forward dispatch.

#include "bwd_dispatch_common.hpp"

void launch_bwd_fp16(const BwdLaunchArgs &args) {
    // Only half variants are instantiated here (TND/BSND). The bf16 variants
    // live in bwd_dispatch_bf16.cpp.
    if (args.kInputLayout == TND) {
        bwd_dispatch_run<half, TND>(args);
    } else {
        bwd_dispatch_run<half, BSND>(args);
    }
}
