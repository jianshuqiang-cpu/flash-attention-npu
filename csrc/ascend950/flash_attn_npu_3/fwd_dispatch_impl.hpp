/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FA950_FWD_DISPATCH_IMPL_HPP
#define FA950_FWD_DISPATCH_IMPL_HPP

#include "autogen/fwd_dispatch_variants.hpp"

// This TU only selects a launcher; kernel definitions live in the leaf TUs.
template <typename DType, Format kLayout, MaskCategory kMask, bool kDn>
inline void launch_fwd_softcap(const FwdLaunchArgs& a)
{
    if (a.has_softcap) {
        launch_fwd_variant<DType, kLayout, kMask, kDn, true>(a);
    } else {
        launch_fwd_variant<DType, kLayout, kMask, kDn, false>(a);
    }
}

template <typename DType, Format kLayout>
void launch_fwd_impl(const FwdLaunchArgs& a)
{
    if (a.mask_category == MaskCategory::NO_MASK) {
        if (a.enable_dn) {
            launch_fwd_softcap<DType, kLayout, MaskCategory::NO_MASK, true>(a);
        } else {
            launch_fwd_softcap<DType, kLayout, MaskCategory::NO_MASK, false>(a);
        }
    } else if (a.mask_category == MaskCategory::MASK_CAUSAL) {
        launch_fwd_softcap<DType, kLayout, MaskCategory::MASK_CAUSAL, false>(a);
    } else {
        launch_fwd_softcap<DType, kLayout, MaskCategory::MASK_SWA, false>(a);
    }
}

#endif
