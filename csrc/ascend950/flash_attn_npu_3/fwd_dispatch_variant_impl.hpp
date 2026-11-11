/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef FA950_FWD_DISPATCH_VARIANT_IMPL_HPP
#define FA950_FWD_DISPATCH_VARIANT_IMPL_HPP

#include <type_traits>
#include "autogen/fwd_dispatch_variants.hpp"
#include "fwd_kernel.cpp"

template <typename DType, Format kLayout, MaskCategory kMask, bool kDn, bool kSoftcap,
          bool kLse, bool kPaged>
inline void launch_fwd_variant_case(const FwdLaunchArgs& a)
{
    static_assert(!kDn || kMask == MaskCategory::NO_MASK);
    constexpr auto cache = kPaged ? CacheMode::pagedCache : CacheMode::normalCache;
    constexpr auto page = kPaged ? PageShape::BnBsND : PageShape::normalShape;
    if constexpr (kDn) {
        FAInferDn<DType, float, kLayout, kLayout, cache, page, kMask, CacheLayout::nd, kLse, kSoftcap>
            <<<a.block_dim, nullptr, a.stream>>>(
                a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device,
                a.o_device, a.lse_device, a.q_seq_device, a.kv_seq_device,
                a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,
                a.tiling_device, a.scale_value, a.softcap_value, a.num_blocks,
                a.max_num_blocks_per_batch, a.kv_seqlen_allocated, a.storage);
    } else {
        FAInfer<DType, float, kLayout, kLayout, cache, page, kMask, CacheLayout::nd, kLse, kSoftcap>
            <<<a.block_dim, nullptr, a.stream>>>(
                a.q_device, a.k_device, a.v_device, a.mask_device, a.block_table_device,
                a.o_device, a.lse_device, a.q_seq_device, a.kv_seq_device,
                a.seq_used_q_device, a.seq_used_kv_device, a.workspace_device,
                a.tiling_device, a.scale_value, a.softcap_value, a.num_blocks,
                a.max_num_blocks_per_batch, a.kv_seqlen_allocated, a.storage);
    }
}

// LSE and cache mode select four kernel entries per heavy TU instead of 32.
// Device kernel templates and their argument lists are unchanged.
template <typename DType, Format kLayout, MaskCategory kMask, bool kDn, bool kSoftcap>
void launch_fwd_variant(const FwdLaunchArgs& a)
{
    if (a.lse_mode) {
        if (a.paged_kv) {
            launch_fwd_variant_case<DType, kLayout, kMask, kDn, kSoftcap, true, true>(a);
        } else {
            launch_fwd_variant_case<DType, kLayout, kMask, kDn, kSoftcap, true, false>(a);
        }
    } else {
        if (a.paged_kv) {
            launch_fwd_variant_case<DType, kLayout, kMask, kDn, kSoftcap, false, true>(a);
        } else {
            launch_fwd_variant_case<DType, kLayout, kMask, kDn, kSoftcap, false, false>(a);
        }
    }
}

#endif
