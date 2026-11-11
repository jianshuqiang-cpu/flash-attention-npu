/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */

#ifndef FA950_BWD_DISPATCH_IMPL_HPP
#define FA950_BWD_DISPATCH_IMPL_HPP

#include "autogen/bwd_dispatch_variants.hpp"

// This TU only selects a launcher; kernel definitions live in the leaf TUs.
template <typename DType, uint32_t kInputLayout>
void bwd_dispatch_run(const BwdLaunchArgs& a)
{
    if (a.has_attn_mask) {
        if (a.deterministic) {
            if (a.is_softcap) {
                bwd_dispatch_variant<DType, kInputLayout, true, true, true>(a);
            } else {
                bwd_dispatch_variant<DType, kInputLayout, true, true, false>(a);
            }
        } else {
            if (a.is_softcap) {
                bwd_dispatch_variant<DType, kInputLayout, true, false, true>(a);
            } else {
                bwd_dispatch_variant<DType, kInputLayout, true, false, false>(a);
            }
        }
    } else {
        if (a.deterministic) {
            if (a.is_softcap) {
                bwd_dispatch_variant<DType, kInputLayout, false, true, true>(a);
            } else {
                bwd_dispatch_variant<DType, kInputLayout, false, true, false>(a);
            }
        } else {
            if (a.is_softcap) {
                bwd_dispatch_variant<DType, kInputLayout, false, false, true>(a);
            } else {
                bwd_dispatch_variant<DType, kInputLayout, false, false, false>(a);
            }
        }
    }
}

#endif // FA950_BWD_DISPATCH_IMPL_HPP
