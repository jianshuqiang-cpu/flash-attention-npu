// Copyright (c) 2026, Minghua Shen.
//
// fp16 instantiation of the v2 forward FAInfer dispatch: 10 half variants
// (BSND: 6 incl. flash-decode; TND: 4). Compiled in its own translation unit so
// it runs in parallel with the bf16 half.

#include "fwd_dispatch_impl.hpp"

template void launch_fwd_impl<half, false>(const FwdLaunchArgs &);  // BSND
template void launch_fwd_impl<half, true>(const FwdLaunchArgs &);   // TND
