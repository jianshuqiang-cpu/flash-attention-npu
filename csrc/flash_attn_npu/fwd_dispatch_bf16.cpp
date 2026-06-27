// Copyright (c) 2026, Minghua Shen.
//
// bf16 instantiation of the v2 forward FAInfer dispatch: 10 bfloat16_t variants
// (BSND: 6 incl. flash-decode; TND: 4). Compiled in its own translation unit so
// it runs in parallel with the fp16 half.

#include "fwd_dispatch_impl.hpp"

template void launch_fwd_impl<bfloat16_t, false>(const FwdLaunchArgs &);  // BSND
template void launch_fwd_impl<bfloat16_t, true>(const FwdLaunchArgs &);   // TND
