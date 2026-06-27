// Copyright (c) 2026, Minghua Shen.
//
// bf16 half of the backward FAGGeneral dispatch: the 32 bfloat16_t
// instantiations (mask x deterministic x headdim x BSND/TND). Compiled in its
// own translation unit so it runs in parallel with the fp16 half and the forward
// dispatch.

#include "bwd_dispatch_common.hpp"

void launch_bwd_bf16(const BwdLaunchArgs &args) {
    // Only bfloat16_t variants are instantiated here (TND/BSND). The fp16
    // variants live in bwd_dispatch_fp16.cpp.
    if (args.kInputLayout == TND) {
        bwd_dispatch_run<bfloat16_t, TND>(args);
    } else {
        bwd_dispatch_run<bfloat16_t, BSND>(args);
    }
}
