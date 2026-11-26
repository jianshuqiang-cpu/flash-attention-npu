// Copyright (c) 2026, Minghua Shen.

#include "fwd_dispatch_impl.hpp"

template void launch_paired_prefill_layout<Format::TND, half, false>(const FwdLaunchArgs&);
template void launch_paired_prefill_layout<Format::TND, bfloat16_t, false>(const FwdLaunchArgs&);
template void launch_paired_prefill_layout<Format::TND, bfloat16_t, true>(const FwdLaunchArgs&);
template void launch_paired_prefill_layout<Format::BSND, bfloat16_t, true>(const FwdLaunchArgs&);
template void launch_paired_prefill_layout<Format::BNSD, bfloat16_t, true>(const FwdLaunchArgs&);
template void launch_grouped_prefill<half>(const FwdLaunchArgs&);
template void launch_grouped_prefill<bfloat16_t>(const FwdLaunchArgs&);
