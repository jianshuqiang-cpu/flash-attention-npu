/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 *
 *
 *   ✅ FP16 / BF16
 *   ✅ Causal mask
 *   ✅ Paged KV (page_table)
 *   ✅ MQA / GQA
 *   ✅ Varlen Q (cu_seqlens_q + max_seqlen_q)
 *   ❌ return_softmax_lse (lse always emitted; wrapper drops it on demand)
 *   ❌ SWA / window_size != (-1, -1)
 *   ❌ num_splits > 1 (FlashDecode)
 *   ❌ pack_gqa, scheduler_metadata, leftpad_k
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <c10/core/Device.h>
#include <torch/extension.h>

#include "acl/acl.h"
#include "fai_host_api.hpp"
#include "fai_tiling.cpp"
#include "fai_tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_from_tensors.hpp"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using flash_attn_npu_950_v3::SeqlenScratch;
using flash_attn_npu_950_v3::fill_inference_context;

// AnyMask (v1): validate device/dtype/contiguity of a provided mask tensor and
// return its device address. Caller MUST guard with has_value(). Addresses are
// stamped into FAInferTilingData and ride the `tiling` GM_ADDR (design D1).
static uint64_t anymask_addr(const std::optional<at::Tensor>& t, const char* name,
                             c10::ScalarType dtype) {
    const at::Tensor& tt = *t;
    TORCH_CHECK(tt.device().type() == at::kPrivateUse1, name, " must be on NPU");
    TORCH_CHECK(tt.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tt.dtype() == dtype, name, " dtype mismatch");
    return reinterpret_cast<uint64_t>(tt.data_ptr());
}

// AnyMask (v1, design D4 host-materialization): build a dense
// [B, maxQSeqlen, maxKvSeqlen] uint8 mask (1 = masked, 0 = kept -- same
// convention as the causal triu matrix consumed by ComputeScaleAndMaxMask) from
// the 7 AnyMask tensors, per AnyMask.md pseudocode. The kernel then loads this
// dense mask tile-by-tile via the existing epilogue path (no kernel-side vector
// mask generation). v1 trades mask memory/host-time for blind-write safety.
// NOTE: O(B*Sq*Skv*Hn) host loop; acceptable for v1, optimize later.
[[maybe_unused]] static at::Tensor materialize_anymask_dense_mask(
    const std::optional<at::Tensor>& hole_num_,
    const std::optional<at::Tensor>& tile_range_,
    const std::optional<at::Tensor>& sparse_compute_,
    const std::optional<at::Tensor>& sparse_mask_,
    const std::optional<at::Tensor>& maskr_,
    const std::optional<at::Tensor>& holel_,
    const std::optional<at::Tensor>& holes_,
    int batch_size, int max_q_seqlen, int max_kv_seqlen, int Hn)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    const int Tq = (max_q_seqlen + TILE_M - 1) / TILE_M;
    const int Tk = (max_kv_seqlen + TILE_N - 1) / TILE_N;
    const int Wk = (Tk + 31) / 32;

    auto to_cpu = [](const std::optional<at::Tensor>& t) -> at::Tensor {
        return t.has_value() ? t->to(at::Device(at::kCPU)) : at::Tensor();
    };
    at::Tensor tile_range_cpu = to_cpu(tile_range_);
    at::Tensor sparse_compute_cpu = to_cpu(sparse_compute_);
    at::Tensor sparse_mask_cpu = to_cpu(sparse_mask_);
    at::Tensor maskr_cpu = to_cpu(maskr_);
    at::Tensor holel_cpu = to_cpu(holel_);
    at::Tensor holes_cpu = to_cpu(holes_);
    const int32_t* p_tr  = tile_range_.has_value()    ? tile_range_cpu.data_ptr<int32_t>()    : nullptr;
    const int32_t* p_sc  = sparse_compute_.has_value()? sparse_compute_cpu.data_ptr<int32_t>(): nullptr;
    const int32_t* p_sm  = sparse_mask_.has_value()   ? sparse_mask_cpu.data_ptr<int32_t>()   : nullptr;
    const int32_t* p_mr  = maskr_.has_value()         ? maskr_cpu.data_ptr<int32_t>()         : nullptr;
    const int32_t* p_hl  = holel_.has_value()         ? holel_cpu.data_ptr<int32_t>()         : nullptr;
    const int32_t* p_hs  = holes_.has_value()         ? holes_cpu.data_ptr<int32_t>()         : nullptr;

    at::Tensor dense = at::zeros({batch_size, max_q_seqlen, (max_kv_seqlen + 127) / 128 * 128},
                                 at::device(c10::kCPU).dtype(at::kByte));
    const int padded_kv = (max_kv_seqlen + 127) / 128 * 128;
    uint8_t* p_dense = dense.data_ptr<uint8_t>();

    for (int b = 0; b < batch_size; ++b) {
        for (int q = 0; q < max_q_seqlen; ++q) {
            const int tq = q / TILE_M;
            for (int k = 0; k < max_kv_seqlen; ++k) {
                const int tk = k / TILE_N;
                bool masked = false;
                // 1) tile_range: k > tile_range[b, tq] -> masked (pseudocode)
                if (!masked && p_tr) {
                    if (k > p_tr[b * Tq + tq]) { masked = true; }
                }
                // 2) sparse_compute: bit=1 -> whole block masked
                if (!masked && p_sc) {
                    int32_t word = p_sc[(b * Tq + tq) * Wk + (tk / 32)];
                    if (word & (1 << (tk % 32))) { masked = true; }
                }
                // 3) sparse_mask: bit=1 -> fine mask via maskr/holel/holes
                if (!masked && p_sm) {
                    int32_t word = p_sm[(b * Tq + tq) * Wk + (tk / 32)];
                    if (word & (1 << (tk % 32))) {
                        if (p_mr && k >= p_mr[b * max_q_seqlen + q]) { masked = true; }
                        if (!masked && p_hl && p_hs) {
                            const int rowBase = (b * max_q_seqlen + q) * Hn;
                            for (int i = 0; i < Hn; ++i) {
                                int32_t hl = p_hl[rowBase + i];
                                int32_t hs = p_hs[rowBase + i];
                                if (hs > 0 && hl <= k && k < hl + hs) { masked = true; break; }
                            }
                        }
                    }
                }
                p_dense[(b * max_q_seqlen + q) * padded_kv + k] = masked ? 1 : 0;
            }
        }
    }
    return dense.to(at::Device(at::kPrivateUse1));
}

std::vector<at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        std::optional<at::Tensor> k_new_,
        std::optional<at::Tensor> v_new_,
        std::optional<at::Tensor> q_v_,
        std::optional<at::Tensor> out_,
        std::optional<at::Tensor> cu_seqlens_q_,
        std::optional<at::Tensor> cu_seqlens_k_,
        std::optional<at::Tensor> cu_seqlens_k_new_,
        std::optional<at::Tensor> seqused_q_,
        std::optional<at::Tensor> seqused_k_,
        std::optional<int64_t>    max_seqlen_q_,
        std::optional<int64_t>    max_seqlen_k_,
        std::optional<at::Tensor> page_table_,
        std::optional<at::Tensor> kv_batch_idx_,
        std::optional<at::Tensor> leftpad_k_,
        std::optional<at::Tensor> rotary_cos_,
        std::optional<at::Tensor> rotary_sin_,
        std::optional<at::Tensor> seqlens_rotary_,
        std::optional<at::Tensor> q_descale_,
        std::optional<at::Tensor> k_descale_,
        std::optional<at::Tensor> v_descale_,
        std::optional<float>      softmax_scale_,
        bool                      is_causal,
        int64_t                   window_size_left,
        int64_t                   window_size_right,
        int64_t                   attention_chunk,
        float                     softcap,
        bool                      is_rotary_interleaved,
        std::optional<at::Tensor> scheduler_metadata_,
        int64_t                   num_splits,
        std::optional<bool>       pack_gqa_,
        int64_t                   sm_margin,
        // AnyMask (v1, 950 forward only). All optional; any non-null enables AnyMask.
        std::optional<at::Tensor> hole_num_,
        std::optional<at::Tensor> tile_range_,
        std::optional<at::Tensor> sparse_compute_,
        std::optional<at::Tensor> sparse_mask_,
        std::optional<at::Tensor> maskr_,
        std::optional<at::Tensor> holel_,
        std::optional<at::Tensor> holes_)
{
    // ============================================================
    // 0. Device guard + stream + AIC core count
    // ============================================================
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // ============================================================
    // 1. dtype + stride sanity
    // ============================================================
    auto q_dtype = q.dtype();
    const bool is_bf16 = (q_dtype == torch::kBFloat16);
    const bool is_fp16 = (q_dtype == torch::kFloat16);
    TORCH_CHECK(is_bf16 || is_fp16,
                "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor v must have contiguous last dimension");

    // ============================================================
    // 2. reject list
    // ============================================================
    TORCH_CHECK(!leftpad_k_.has_value(),
                "950 backend (v3) does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value()
                && !seqlens_rotary_.has_value(),
                "950 backend (v3) does not support rotary embedding");
    TORCH_CHECK(!q_descale_.has_value() && !k_descale_.has_value()
                && !v_descale_.has_value(),
                "950 backend (v3) does not support FP8 descales");
    TORCH_CHECK(softcap == 0.0f, "950 backend (v3) does not support softcap");
    TORCH_CHECK(window_size_left == -1 && window_size_right == -1,
                "950 backend (v3) does not support SWA");
    TORCH_CHECK(attention_chunk == 0,
                "950 backend (v3) does not support attention_chunk");
    TORCH_CHECK(!scheduler_metadata_.has_value(),
                "950 backend (v3) does not consume scheduler_metadata");
    TORCH_CHECK(num_splits == 0 || num_splits == 1,
                "950 backend (v3) only supports num_splits=0 or 1");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(),
                "950 backend (v3) does not support pack_gqa");

    // ============================================================
    // 3. paged / varlen mode + per-tensor checks
    // ============================================================
    const bool paged_KV    = page_table_.has_value();
    const bool is_varlen_q = cu_seqlens_q_.has_value();
    const bool is_varlen_kv = cu_seqlens_k_.has_value();

    // ---- AnyMask (v1): presence + mutual-exclusion / scope rejects ----
    const bool anyMaskPresent = hole_num_.has_value() || tile_range_.has_value() ||
        sparse_compute_.has_value() || sparse_mask_.has_value() || maskr_.has_value() ||
        holel_.has_value() || holes_.has_value();
    if (anyMaskPresent) {
        TORCH_CHECK(!is_causal,
                    "AnyMask is mutually exclusive with is_causal on 950 (v3) v1");
        TORCH_CHECK(!is_varlen_q,
                    "AnyMask v1 only supports BSND (non-varlen) on 950 (v3)");
        TORCH_CHECK(!paged_KV,
                    "AnyMask v1 only supports non-paged KV on 950 (v3)");
        TORCH_CHECK(!is_bf16,
                    "AnyMask v1 only supports fp16 high_prec on 950 (v3)");
    }

    at::Tensor cu_seqlens_q, cu_seqlens_k, page_table, seqlens_k;

    if (paged_KV) {
        page_table = page_table_.value();
        TORCH_CHECK(page_table.dtype() == torch::kInt32,
                    "page_table must have dtype int32");
        TORCH_CHECK(page_table.stride(-1) == 1,
                    "page_table must have contiguous last dimension");
    }
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.device().type() == at::kPrivateUse1,
                    "cu_seqlens_q must be on NPU");
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
                    "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(),
                    "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.device().type() == at::kPrivateUse1,
                    "cu_seqlens_k must be on NPU");
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
                    "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV,
                    "If cu_seqlens_k is passed in, paged table is not supported");
    }
    TORCH_CHECK(seqused_k_.has_value(),
                "950 backend (v3) requires seqused_k (per-batch KV seqlen) — the "
                "Python wrapper passes cache_seqlens through this argument");
    seqlens_k = seqused_k_.value();
    CHECK_CONTIGUOUS(seqlens_k);
    TORCH_CHECK(seqlens_k.device().type() == at::kPrivateUse1,
                "seqused_k must be on NPU");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");

    // ============================================================
    // 4. Output tensor
    // ============================================================
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype,
                    "output must have the same dtype as inputs");
        TORCH_CHECK(out.stride(-1) == 1,
                    "Output tensor must have contiguous last dimension");
    } else {
        out = torch::empty_like(q);
    }

    // ============================================================
    // 5. Shape extraction
    // ============================================================
    const auto sizes = q.sizes();
    int batch_size, seqlen_q, num_heads, head_size_q;
    if (is_varlen_q) {
        batch_size = static_cast<int>(cu_seqlens_q.size(0)) - 1;
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
        num_heads = static_cast<int>(sizes[1]);
        head_size_q = static_cast<int>(sizes[2]);
    } else {
        batch_size = static_cast<int>(sizes[0]);
        seqlen_q = static_cast<int>(sizes[1]);
        num_heads = static_cast<int>(sizes[2]);
        head_size_q = static_cast<int>(sizes[3]);
    }
    const int max_num_blocks_per_seq = !paged_KV ? 0 : static_cast<int>(page_table.size(1));
    const int num_blocks = !paged_KV ? 0 : static_cast<int>(k.size(0));
    const int page_block_size = !paged_KV ? 128 : static_cast<int>(k.size(1));
    const int num_heads_k = static_cast<int>(k.size(2));
    const int head_size_v = static_cast<int>(v.size(-1));

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(!(head_size_q != 64 && head_size_q != 128),
                "FlashAttention only supports q head dimension 64 or 128");
    TORCH_CHECK(!(head_size_v != 64 && head_size_v != 128),
                "FlashAttention only supports v head dimension 64 or 128");
    TORCH_CHECK(!(page_block_size != 128 && page_block_size != 256 && page_block_size != 512 && page_block_size != 1024),
                "FlashAttention only supports page_block_size dimension 128 or 256 or 512 or 1024");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    // ============================================================
    // 6. Pull cu_seqlens_q / seqused_k to host as int32 — the 950
    //    FAInferContext consumes int64 lists, so we widen on host.
    // ============================================================
    at::Tensor cu_seqlen_q_cpu;
    if (is_varlen_q) {
        cu_seqlen_q_cpu = cu_seqlens_q.to(at::Device(at::kCPU));
    }
    at::Tensor seqlens_k_cpu = seqlens_k.to(at::Device(at::kCPU));

    // ============================================================
    // 7. Build FAInferContext + run host-side tiling
    // ============================================================
    SeqlenScratch scratch;
    optiling::FAInferContext ctx;
    fill_inference_context(
        ctx, scratch,
        q, k, v,
        is_varlen_q ? &cu_seqlen_q_cpu : nullptr,
        &seqlens_k_cpu,
        paged_KV, page_block_size, num_blocks, max_num_blocks_per_seq,
        is_causal, is_varlen_q, is_bf16,
        batch_size, seqlen_q, num_heads, num_heads_k,
        head_size_q, head_size_v,
        softmax_scale_.value_or(1.0f / std::sqrt(static_cast<float>(head_size_q))),
        /* lse_flag= */ true,
        /* layout_str= */ is_varlen_q ? "TND" : "BSND");

    FAInferTilingData tilingData{};
    {
        optiling::FAInferTiling tiler(ctx);
        tiler.SetCoreNum(blockDim);
        tiler.DoTiling(tilingData);
    }

    // The 950 chunk-prefill driver overrides workSpaceSize to 128 MiB
    constexpr uint64_t WS_FLOOR = uint64_t(1024) * 1024 * 32 * 4;  // 128 MiB
    if (tilingData.workSpaceSize < WS_FLOOR) {
        tilingData.workSpaceSize = WS_FLOOR;
    }

    // ---- AnyMask (v1): validate provided tensors + stamp tiling fields ----
    // Addresses ride the `tiling` GM_ADDR (design D1); anyMaskEnabled selects
    // the AnyMask path at runtime (design D7). Each feature is gated on its own
    // address != 0, so partial provision (e.g. tile_range only) is supported.
    tilingData.anyMaskEnabled = anyMaskPresent ? 1u : 0u;
    tilingData.holeMaxNum = 0u;
    tilingData.holeNumAddr = 0;
    tilingData.tileRangeAddr = 0;
    tilingData.sparseComputeAddr = 0;
    tilingData.sparseMaskAddr = 0;
    tilingData.maskrAddr = 0;
    tilingData.holelAddr = 0;
    tilingData.holesAddr = 0;
    if (anyMaskPresent) {
        constexpr int TILE_M = 128;  // qBaseTile
        constexpr int TILE_N = 128;  // kvBaseTile
        const int Tq = (seqlen_q + TILE_M - 1) / TILE_M;
        const int maxKv = static_cast<int>(tilingData.maxKvSeqlen);
        const int Tk = (maxKv + TILE_N - 1) / TILE_N;
        const int Wk = (Tk + 31) / 32;
        // Hn: from hole_num (preferred) else holel last dim
        int Hn = 0;
        if (hole_num_.has_value()) Hn = static_cast<int>(hole_num_->size(0));
        else if (holel_.has_value()) Hn = static_cast<int>(holel_->size(2));
        constexpr int HN_MAX = 16;  // UB budget (design Open Question 2)
        TORCH_CHECK(Hn <= HN_MAX, "AnyMask v1 requires holeMaxNum (Hn) <= ", HN_MAX,
                    " (UB budget), got ", Hn);
        tilingData.holeMaxNum = static_cast<uint32_t>(Hn);

        if (hole_num_.has_value()) {
            TORCH_CHECK(hole_num_->sizes().size() == 1 && hole_num_->size(0) == Hn,
                        "hole_num must be [Hn] int16");
            tilingData.holeNumAddr = anymask_addr(hole_num_, "hole_num", torch::kInt16);
        }
        if (tile_range_.has_value()) {
            TORCH_CHECK(tile_range_->sizes().size() == 2 &&
                        tile_range_->size(0) == batch_size && tile_range_->size(1) == Tq,
                        "tile_range must be [BatchSize, Tq] int32");
            tilingData.tileRangeAddr = anymask_addr(tile_range_, "tile_range", torch::kInt32);
        }
        if (sparse_compute_.has_value()) {
            TORCH_CHECK(sparse_compute_->sizes().size() == 3 &&
                        sparse_compute_->size(0) == batch_size &&
                        sparse_compute_->size(1) == Tq &&
                        sparse_compute_->size(2) == Wk,
                        "sparse_compute must be [BatchSize, Tq, Wk] int32");
            tilingData.sparseComputeAddr = anymask_addr(sparse_compute_, "sparse_compute", torch::kInt32);
        }
        if (sparse_mask_.has_value()) {
            TORCH_CHECK(sparse_mask_->sizes().size() == 3 &&
                        sparse_mask_->size(0) == batch_size &&
                        sparse_mask_->size(1) == Tq &&
                        sparse_mask_->size(2) == Wk,
                        "sparse_mask must be [BatchSize, Tq, Wk] int32");
            tilingData.sparseMaskAddr = anymask_addr(sparse_mask_, "sparse_mask", torch::kInt32);
        }
        if (maskr_.has_value()) {
            TORCH_CHECK(maskr_->sizes().size() == 2 &&
                        maskr_->size(0) == batch_size && maskr_->size(1) == seqlen_q,
                        "maskr must be [BatchSize, Sq] int32");
            tilingData.maskrAddr = anymask_addr(maskr_, "maskr", torch::kInt32);
        }
        if (holel_.has_value()) {
            TORCH_CHECK(holel_->sizes().size() == 3 &&
                        holel_->size(0) == batch_size &&
                        holel_->size(1) == seqlen_q &&
                        holel_->size(2) == Hn,
                        "holel must be [BatchSize, Sq, Hn] int32");
            tilingData.holelAddr = anymask_addr(holel_, "holel", torch::kInt32);
        }
        if (holes_.has_value()) {
            TORCH_CHECK(holes_->sizes().size() == 3 &&
                        holes_->size(0) == batch_size &&
                        holes_->size(1) == seqlen_q &&
                        holes_->size(2) == Hn,
                        "holes must be [BatchSize, Sq, Hn] int32");
            tilingData.holesAddr = anymask_addr(holes_, "holes", torch::kInt32);
        }
        fprintf(stderr, "[AnyMask][host] anyMaskPresent Hn=%d Tq=%d Tk=%d Wk=%d batch=%d seqlen_q=%d maxKv=%d\n",
                Hn, Tq, Tk, Wk, batch_size, seqlen_q, maxKv);
        fprintf(stderr, "[AnyMask][host] addrs holeNum=%p tileRange=%p sparseCompute=%p sparseMask=%p maskr=%p holel=%p holes=%p\n",
                (void*)tilingData.holeNumAddr, (void*)tilingData.tileRangeAddr,
                (void*)tilingData.sparseComputeAddr, (void*)tilingData.sparseMaskAddr,
                (void*)tilingData.maskrAddr, (void*)tilingData.holelAddr,
                (void*)tilingData.holesAddr);
    }

    // ============================================================
    // 8. Allocate output-side buffers on NPU
    // ============================================================
    auto workspace = at::empty(
        {static_cast<int64_t>(tilingData.workSpaceSize)},
        at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor softmaxlse;
    if (is_varlen_q) {
        // Match v3's varlen lse shape: {total_q, num_heads}
        softmaxlse = at::empty({sizes[0], num_heads},
                               at::device(at::kPrivateUse1).dtype(at::kFloat));
    } else {
        softmaxlse = at::empty({batch_size, seqlen_q, num_heads},
                               at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    // ============================================================
    // 9. Tiling host→device (CPU byte tensor + .to(kPrivateUse1) —
    //    same idiom
    // ============================================================
    at::Tensor tiling_cpu = at::empty(
        {static_cast<int64_t>(sizeof(FAInferTilingData))},
        at::device(c10::kCPU).dtype(at::kByte));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &tilingData,
                sizeof(FAInferTilingData));
    at::Tensor tiling_dev = tiling_cpu.to(at::Device(at::kPrivateUse1));

    // ============================================================
    // 10. Build kernelKey + launch via fai_host::LaunchFAI
    // ============================================================
    const Format fmt = is_varlen_q ? Format::TND : Format::BSND;
    const CacheMode cacheMode = paged_KV ? CacheMode::pagedCache
                                           : CacheMode::normalCache;
    const PageShape pageShape = paged_KV ? PageShape::BnBsND
                                           : PageShape::normalShape;
    const uint32_t maskTypeKey = is_causal ? 1u : 0u;
    const uint32_t innerPrec = 0u; // FP32 accum
    const std::string dataType = is_bf16 ? "bf16" : "half";
    const std::string cacheLayout = "nd"; // nd only

    const uint32_t kernelKey = fai_host::BuildKernelKey(
        dataType, cacheLayout, maskTypeKey, innerPrec,
        fmt, cacheMode, pageShape);

    // device pointers
    auto qDev = static_cast<uint8_t*>(q.data_ptr());
    auto kDev = static_cast<uint8_t*>(k.data_ptr());
    auto vDev = static_cast<uint8_t*>(v.data_ptr());
    auto oDev = static_cast<uint8_t*>(out.data_ptr());
    auto lseDev = static_cast<uint8_t*>(softmaxlse.data_ptr());
    auto wsDev = static_cast<uint8_t*>(workspace.data_ptr());
    auto tilDev = static_cast<uint8_t*>(tiling_dev.data_ptr());

    const auto i64_npu = at::device(at::kPrivateUse1).dtype(at::kLong);
    at::Tensor q_seq_i64 = is_varlen_q
        ? cu_seqlens_q
        : at::empty({batch_size}, i64_npu);
    at::Tensor kv_seq_i64 = is_varlen_kv
        ? cu_seqlens_k
        : seqlens_k;
    auto qSeqDev  = static_cast<uint8_t*>(q_seq_i64.data_ptr());
    auto kvSeqDev = static_cast<uint8_t*>(kv_seq_i64.data_ptr());
    auto blockTableDev = paged_KV
        ? static_cast<uint8_t*>(page_table.data_ptr())
        : nullptr;
    uint8_t* maskDev = nullptr;
    at::Tensor mask_npu_tensor;
    at::Tensor mask_cpu_tensor;
    if (anyMaskPresent) {
        // AnyMask (design D4 kernel-side vectorized): kernel reads maskr/holel/
        // holes + sparse_compute/sparse_mask bitmaps directly from tiling addrs;
        // no host dense mask. maskDev unused for AnyMask (nullptr).
        maskDev = nullptr;
    } else if (is_causal) {
        mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDev = static_cast<uint8_t*>(mask_npu_tensor.data_ptr());
    }

    const bool enableDN =
        (!is_causal) && (head_size_q <= 128) && (head_size_v <= 128) && (innerPrec == 0u);

    fprintf(stderr, "[AnyMask][host] launch kernelKey=%u enableDN=%d blockDim=%u anyMaskEnabled=%u holeMaxNum=%u\n",
            kernelKey, (int)enableDN, blockDim, tilingData.anyMaskEnabled, tilingData.holeMaxNum);
    const aclError err = fai_host::LaunchFAI(
        kernelKey, enableDN,
        blockDim, aclStream,
        qDev, kDev, vDev, maskDev, blockTableDev,
        oDev, lseDev, qSeqDev, kvSeqDev,
        wsDev, tilDev);
    fprintf(stderr, "[AnyMask][host] LaunchFAI returned err=%d (0=SUCCESS)\n", (int)err);
    TORCH_CHECK(err == ACL_SUCCESS,
                "950 backend (v3): unsupported kernelKey=", kernelKey,
                " (no launcher registered for "
                "dtype=", dataType, " cacheLayout=", cacheLayout,
                " maskType=", maskTypeKey, " innerPrec=", innerPrec,
                " layout=", (fmt == Format::TND ? "TND" : "BSND"),
                " cacheMode=", (paged_KV ? "paged" : "normal"),
                ")");
    const aclError sync_err = aclrtSynchronizeStream(aclStream);
    fprintf(stderr, "[AnyMask][host] aclrtSynchronizeStream sync_err=%d (0=SUCCESS; nonzero=kernel CRASHED; if this line never appears the kernel HUNG=deadlock)\n", (int)sync_err);
    TORCH_CHECK(sync_err == ACL_SUCCESS,
                "950 backend (v3): aclrtSynchronizeStream failed after LaunchFAI, err=",
                sync_err);

    at::Tensor empty_accum = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    return {out, softmaxlse, empty_accum, empty_accum};
}
