// Copyright (c) 2026, Minghua Shen.
//
// bf16 instantiation of the v2 varlen backward FAGVarlenOpt dispatch.
// Compiled in its own translation unit so it runs in parallel with the fp16 half.

#include "varlen_bwd_dispatch_impl.hpp"

template void launch_varlen_bwd_impl<bfloat16_t>(const VarlenBwdLaunchArgs &);
