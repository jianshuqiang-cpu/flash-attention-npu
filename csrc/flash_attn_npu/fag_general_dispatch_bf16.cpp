// Copyright (c) 2026, Minghua Shen.
//
// bf16 instantiation of the v2 FAGGeneral backward dispatch. Compiled in its
// own translation unit so it runs in parallel with the fp16 half. Instantiates
// the bf16 FAGGeneral variants for both layouts (TND + BSND) => 32 kernels.

#include "fag_general_dispatch_impl.hpp"

template void launch_fag_general_dispatch_bf16<TND>(const FagGeneralLaunchArgs &);
template void launch_fag_general_dispatch_bf16<BSND>(const FagGeneralLaunchArgs &);
