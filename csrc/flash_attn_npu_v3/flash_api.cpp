// =============================================================================
// flash_api.cpp — FlashAttention v3 的 C++ pybind11 入口文件
// =============================================================================
// 本文件是 v3 版本 FlashAttention 在 Ascend NPU 上的宿主端（Host）入口。
// v3 与 v2 的核心区别：v3 将标准前向、KV-Cache 前向、变长（varlen）前向
// 统一合并为单个 mha_fwd 函数，通过可选参数（page_table / cu_seqlens 等）
// 来区分不同使用场景，接口更简洁但参数更多。
//
// 本文件主要职责：
//   1. 接收 PyTorch 张量参数并做输入校验（dtype / 连续性 / 维度）
//   2. 计算 Tiling（分块）参数并填充 FAInferTilingData 结构体
//   3. 计算并分配 Device 端 workspace 显存
//   4. 根据数据类型/掩码/布局组合，选择对应的模板实例化并启动 AscendC kernel
//   5. 通过 PYBIND11_MODULE 将 mha_fwd 注册为 Python 可调用算子 "fwd"
// =============================================================================

#include <torch/extension.h>

#include "mha_fwd_kvcache.cpp"          // AscendC kernel 实现（SplitFuse::FAInfer 模板）
#include "tilingdata.h"                  // FAInferTilingData 结构体定义（Host↔Device 传递的分块参数）
#include "torch_npu/csrc/core/npu/NPUStream.h"  // NPU 流（stream）管理
#include "acl/acl.h"                     // Ascend Computing Language 基础接口
#include "runtime/rt_ffts.h"             // FFTS（Fast Task Schedule）多核任务调度接口
#include "kernel_common.hpp"            // kernel 通用常量与枚举（Q_TILE_CEIL / MaskType / inputLayout 等）
#include "kernel_operator.h"            // AscendC kernel 算子编程框架
#include "tiling/platform/platform_ascendc.h"  // 平台信息查询（AI Core 数量等）

// 宏：校验张量在内存中是否连续，不连续则抛出异常
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// -----------------------------------------------------------------------------
// GetQNBlockTile — 计算每个 tile 在 N（head 头数）维度上的分块大小
// -----------------------------------------------------------------------------
// 在 GQA/MQA 场景下，query 头数 num_heads 是 KV 头数 num_heads_k 的 groupSize 倍。
// 本函数决定：一个 tile 内同时处理多少个 query 头（即 N 维度的分块粒度）。
//
// 计算逻辑：
//   1. qRowNumCeil = Q_TILE_CEIL(128)，即单个 tile 最多容纳 128 行 query
//   2. 若 qSeqlen 不为 0，则每个头占 qSeqlen 行，一个 tile 能容纳
//      (128 / qSeqlen) 个头；再按 N_SPLIT_HELPER(2) 对齐向下取整，
//      保证 N 维分块是 2 的倍数（与 Cube/Vector 双引擎流水线对齐）
//   3. 结果不超过 groupSize（GQA 分组数），也不小于 1
//
// 参数：
//   qSeqlen   — 当前 batch 的 query 序列长度
//   groupSize — GQA 分组数 = num_heads / num_heads_k
// 返回：N 维度每个 tile 处理的 query 头数
// -----------------------------------------------------------------------------
uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    // 计算一个 tile 能容纳多少个头，并按 N_SPLIT_HELPER 对齐
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    // 不超过 GQA 分组数 groupSize
    qNBlockTile = std::min(qNBlockTile, groupSize);
    // 至少为 1
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

// -----------------------------------------------------------------------------
// GetQSBlockTile — 计算每个 tile 在 S（query 序列）维度上的分块大小
// -----------------------------------------------------------------------------
// 决定一个 tile 内处理多少行 query（S 维度的分块粒度）。
// 当前实现固定返回 Q_TILE_CEIL(128)，即每个 tile 处理 128 行 query。
// kvSeqlen 参数为预留接口，便于未来根据 KV 序列长度动态调整。
//
// 参数：
//   kvSeqlen — 当前 batch 的 KV 序列长度（当前未使用）
// 返回：S 维度每个 tile 处理的 query 行数（固定 128）
// -----------------------------------------------------------------------------
uint32_t GetQSBlockTile(int64_t kvSeqlen)
{
    uint32_t qSBlockTile = Q_TILE_CEIL;
    return qSBlockTile;
}

// =============================================================================
// mha_fwd — FlashAttention v3 统一前向推理入口（Host 端）
// =============================================================================
// 本函数是 v3 的核心入口，统一处理以下场景（通过可选参数区分）：
//   • 标准注意力：q/k/v 均为 (b, s, h, d) 的 BSND 布局
//   • 变长注意力（varlen）：提供 cu_seqlens_q / cu_seqlens_k，使用 TND 布局
//   • KV-Cache 分页注意力（paged KV）：提供 page_table，k/v 为分页块布局
//   • 因果掩码（causal mask）：is_causal=true 时施加下三角掩码
//
// 整体流程：
//   1. 输入校验（dtype / 连续性 / 不支持的特性检查）
//   2. 提取维度信息（batch / seqlen / heads / head_size）
//   3. 计算 workspace 大小并分配 Device 显存
//   4. 填充 FAInferTilingData 分块参数
//   5. 逐 batch 计算任务总数（totalTaskNum）
//   6. 生成 causal mask（若需要）
//   7. 获取 FFTS 多核调度地址
//   8. 根据 dtype×pagedKV×causal×varlen 组合选择模板，启动 kernel
//   9. 返回输出 out 和 logsumexp（softmaxlse）
//
// 返回值：{out, softmaxlse, out_accum, softmax_lse_accum}
//   其中 out_accum 和 softmax_lse_accum 当前未使用（预留多 split 合并场景）
// =============================================================================
std::vector<at::Tensor>
mha_fwd(at::Tensor q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
                at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
                std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
                std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
                std::optional<at::Tensor> q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
                std::optional<at::Tensor> out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
                // cu_seqlens_q_[i] 表示第 i 个样本之前所有样本的 token 总数（含 i=0 时为 0）。
                std::optional<at::Tensor> cu_seqlens_q_,  // b+1
                std::optional<at::Tensor> cu_seqlens_k_,  // b+1
                std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
                std::optional<at::Tensor> seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
                std::optional<at::Tensor> seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
                std::optional<int64_t> max_seqlen_q_,
                // TODO: check if we need max_seqlen_k
                std::optional<int64_t> max_seqlen_k_,
                // FlashAttention 的Paged KV-Cache模式入参，用于在推理场景下支持类似 vLLM 的非连续 KV 存储。
                std::optional<at::Tensor> page_table_, // (b_k, max_num_pages_per_seq)
                std::optional<at::Tensor> kv_batch_idx_, // b. indices to index into the KV cache
                std::optional<at::Tensor> leftpad_k_, // b
                std::optional<at::Tensor> rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<at::Tensor> rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<at::Tensor> seqlens_rotary_, // b
                std::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
                std::optional<at::Tensor> k_descale_,  // (b, h_k)
                std::optional<at::Tensor> v_descale_,  // (b, h_k)
                std::optional<float> softmax_scale_,
                bool is_causal,
                int64_t window_size_left,
                int64_t window_size_right,
                int64_t attention_chunk,
                float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
                int64_t num_splits,
                std::optional<bool> pack_gqa_,
                int64_t sm_margin
                )
{
    // 获取当前 NPU 设备并切换到 q 所在设备，获取 NPU 计算流
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // ---- 数据类型校验：仅支持 FP16 和 BF16 ----
    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;
    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    // ---- 连续性校验：q/k/v 最后一维必须连续（NPU Cube 矩阵乘要求） ----
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor v must have contiguous last dimension");
    // 在 CPU 上分配 1024 字节缓冲区，用于存放 FAInferTilingData 分块参数
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));

    // 将 CPU 缓冲区重解释为 FAInferTilingData 指针，后续通过 setter 填充各字段
    FAInferTilingData* tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    // 查询当前 NPU 的 AI Core 数量，作为 kernel 启动的 block 数（多核并行度）
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k_, v_, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;

    at::Tensor cu_seqlens_q, cu_seqlens_k;
    at::Tensor out_accum, softmax_lse_accum;
    float softmax_scale;

    // ---- 场景标志位：根据可选参数判断当前是哪种注意力模式 ----
    const bool paged_KV = page_table_.has_value();        // 是否为分页 KV-Cache 模式
    const bool is_varlen_q = cu_seqlens_q_.has_value();   // query 是否为变长（TND 布局）
    const bool is_varlen_kv = cu_seqlens_k_.has_value();  // KV 是否为变长

    // ---- 分页 KV-Cache 参数校验 ----
    if (paged_KV) {
        auto page_table = page_table_.value();
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    // ---- 变长 query 参数校验 ----
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    // ---- 变长 KV 参数校验（与分页 KV 互斥） ----
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }

    // ---- seqused_k：每个 batch 实际使用的 KV 长度 ----
    if (seqused_k_.has_value()) {
        seqlens_k = seqused_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    }

    // ---- 以下特性当前 NPU 版本暂不支持，传入则报错 ----
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!rotary_sin_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!seqlens_rotary_.has_value(), "NPU FlashAttention does not support seqlens_rotary");
    TORCH_CHECK(!q_descale_.has_value(), "NPU FlashAttention does not support q_descale");
    TORCH_CHECK(!k_descale_.has_value(), "NPU FlashAttention does not support k_descale");
    TORCH_CHECK(!v_descale_.has_value(), "NPU FlashAttention does not support v_descale");
    TORCH_CHECK(softcap == 0.0f, "NPU FlashAttention does not support softcap");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right");
    TORCH_CHECK(attention_chunk == 0, "NPU FlashAttention does not support attention_chunk");
    TORCH_CHECK(!scheduler_metadata_.has_value(), "NPU FlashAttention does not support scheduler_metadata");
    TORCH_CHECK(num_splits == 1 || num_splits == 0, "NPU FlashAttention only supports num_splits=1 or num_splits=0");
    TORCH_CHECK(!pack_gqa_.has_value() || !pack_gqa_.value(), "NPU FlashAttention does not support pack_gqa");

    // ---- 解包可选参数到局部变量 ----
    if (is_varlen_kv) {
        cu_seqlens_k = cu_seqlens_k_.value();
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then paged table is not supported");
    }
    if (k_new_.has_value()) {
        k_ = k_new_.value();
    }
    if (v_new_.has_value()) {
        v_ = v_new_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (kv_batch_idx_.has_value()) {
        cache_batch_idx = kv_batch_idx_.value();
    }
    if (paged_KV) {
        block_table = page_table_.value();
    }
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    // ---- 输出张量：若外部未提供则按 q 的形状新建 ----
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "output must have the same dtype as inputs");
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    }  else {
        out = torch::empty_like(q);
    }
    const auto sizes = q.sizes();
    
    // ---- 从 q 的形状提取核心维度信息 ----
    // 变长模式下 q 为 (total_q, h, d)，标准模式下 q 为 (b, s_q, h, d)
    int batch_size = 0;
    int seqlen_q = 0;
    int num_heads = 0;
    int head_size_og = 0;
    if (is_varlen_q) {
        batch_size = cu_seqlens_q.size(0) - 1;  // batch 数 = cu_seqlens 长度 - 1
        seqlen_q = static_cast<int>(max_seqlen_q_.value());
        num_heads = sizes[1];
        head_size_og = sizes[2];
    } else {
        batch_size = sizes[0];
        seqlen_q = sizes[1];
        num_heads = sizes[2];
        head_size_og = sizes[3];
    }
    // 分页 KV 相关维度
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);  // 每个序列最大页数
    const int num_blocks = !paged_KV ? 0 : k.size(0);                        // 总页数
    const int page_block_size = !paged_KV ? 128 : k.size(1);                 // 每页的 KV 序列长度
    const int num_heads_k = k.size(2);                                        // KV 头数（GQA/MQA 下 < num_heads）

    // ---- 维度合法性校验 ----
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // ---- 将 seqlens_k 拷贝到 CPU，用于逐 batch 计算任务数 ----
    at::Tensor seqlenk_cpu_tensor = seqlens_k.to(at::Device(at::kCPU));
    int32_t* seqlens_k_cpu = static_cast<int32_t *>(seqlenk_cpu_tensor.data_ptr());
    int32_t* cu_seqlen_q_cpu = nullptr;
    at::Tensor cu_seqlen_q_cpu_tensor;
    if (is_varlen_q) {
        cu_seqlen_q_cpu_tensor = cu_seqlens_q.to(at::Device(at::kCPU));
        cu_seqlen_q_cpu = static_cast<int32_t *>(cu_seqlen_q_cpu_tensor.data_ptr());
    }
    // ---- 填充 FAInferTilingData 分块参数（Host 端写入，后续拷贝到 Device） ----
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_embeddingSizeV(static_cast<uint32_t>(head_size_og));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    tiling_cpu_ptr->set_maskType(static_cast<uint32_t>(is_causal));  // 0=NO_MASK, 1=MASK_CAUSAL
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    // ---- 计算 workspace 各部分大小 ----
    // workspace 分为 4 个区域：mm1 输出(QK^T)、softmax 在线计算、mm2 输出(PV)、update 缓冲
    // 每个区域大小 = blockDim × WORKSPACE_BLOCK_SIZE_DB(128×512) × 元素大小 × 预取缓冲数(PRELANCH_NUM=3)
    uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 512;
    uint64_t PRELANCH_NUM = 3;
    uint64_t mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;    // QK^T 矩阵乘输出（float32，4 字节）
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        2 * PRELANCH_NUM;    // online softmax 中间结果（half/bf16，2 字节）
    uint64_t mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;    // PV 矩阵乘输出（float32，4 字节）
    uint64_t UpdateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
        4 * PRELANCH_NUM;    // rescale 更新缓冲（float32，4 字节）
    int64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;

    
    // ---- 在 Device 上分配 workspace 显存 ----
    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
    // ---- 分配 softmax logsumexp 输出张量 ----
    // 标准模式: (batch, seqlen_q, num_heads)；变长模式: (total_q, num_heads)
    at::Tensor softmaxlse = at::empty({batch_size, seqlen_q, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    if (is_varlen_q) {
        softmaxlse = at::empty({sizes[0], num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    }
    // 初始化为 +inf（FlashAttention online softmax 的约定：未处理的位置 logsumexp = -inf，故 exp 后为 0）
    softmaxlse.fill_(std::numeric_limits<float>::infinity());
    // 将 workspace 各部分大小写入 tiling 参数，供 kernel 端使用
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    // ---- 逐 batch 计算总任务数（totalTaskNum） ----
    // 每个 batch 的任务数 = N 维分块数 × S 维分块数
    //   N 维分块数 = (groupSize / curQNBlockTile) 向上取整 × num_heads_k
    //   S 维分块数 = (qSeqlen / curQSBlockTile) 向上取整
    // kernel 端通过 blockIdx 和 totalTaskNum 来分配各 AI Core 的工作量
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = num_heads / num_heads_k;  // GQA 分组数
    for (int32_t batchIdx = 0; batchIdx < batch_size; batchIdx++) {
        uint64_t qSeqlen = seqlen_q;
        if (is_varlen_q) {
            // 变长模式：从 cu_seqlens_q 读取当前 batch 的实际 query 长度
            qSeqlen = *(cu_seqlen_q_cpu + batchIdx + 1) - *(cu_seqlen_q_cpu + batchIdx);
        }
        uint64_t kvSeqlen = *(seqlens_k_cpu + batchIdx);  // 当前 batch 的 KV 长度
        uint64_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);  // N 维每 tile 头数
        uint64_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;  // 每组 N 维分块数
        uint64_t curQNBlockNum = qNBlockNumPerGroup * num_heads_k;  // N 维总分块数
        uint64_t curQSBlockTile = GetQSBlockTile(kvSeqlen);  // S 维每 tile 行数（固定 128）
        uint64_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;  // S 维分块数
        uint64_t curTaskNum = curQNBlockNum * curQSBlockNum;  // 当前 batch 任务数
        if (batchIdx == 0) {
            // 记录第一个 batch 的任务数，kernel 端用于计算偏移
            tiling_cpu_ptr->set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    // ---- 生成 causal mask（若 is_causal=true） ----
    // 创建 2048×2048 的上三角矩阵（triu），1 表示需要掩码的位置
    // kernel 端会将这些位置的 attention score 设为 -inf
    at::Tensor mask_gpu_tensor;
    if (is_causal) {
        at::Tensor mask_cpu_tensor = at::empty({2048, 2048}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(at::ones_like(mask_cpu_tensor), 1);
        mask_gpu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
    }
    // ---- 将 tiling 参数从 CPU 拷贝到 Device ----
    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    // ---- 准备 query/KV 序列长度张量（Device 端） ----
    at::Tensor seqlenk_gpu_tensor;
    at::Tensor seqlenq_gpu_tensor;
    if (is_varlen_q) {
        seqlenq_gpu_tensor = cu_seqlens_q;  // 变长模式：使用累积序列长度
    } else {
        seqlenq_gpu_tensor = at::empty({0}, at::device(at::kPrivateUse1).dtype(at::kInt));  // 空张量占位
    }
    if (is_varlen_kv) {
        seqlenk_gpu_tensor = cu_seqlens_k;
    } else {
        seqlenk_gpu_tensor = seqlens_k;
    }
    // ---- 获取 FFTS（Fast Task Schedule）多核任务调度地址 ----
    // FFTS 是 Ascend NPU 的硬件级多核同步机制，用于 Cube/Vector 双引擎流水线同步
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // ---- 获取所有输入/输出张量的 Device 指针 ----
    auto qDevice = static_cast<uint8_t *>(q.data_ptr());
    auto kDevice = static_cast<uint8_t *>(k.data_ptr());
    auto vDevice = static_cast<uint8_t *>(v.data_ptr());
    uint8_t * blockTableDevice = nullptr;
    uint8_t * maskDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t *>(block_table.data_ptr());
    }
    if (is_causal) {
        maskDevice = static_cast<uint8_t *>(mask_gpu_tensor.data_ptr());
    }
    auto oDevice = static_cast<uint8_t *>(out.data_ptr());
    auto qSeqDevice = static_cast<uint8_t *>(seqlenq_gpu_tensor.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t *>(seqlenk_gpu_tensor.data_ptr());
    auto workspaceDevice = static_cast<uint8_t *>(workspace_tensor.data_ptr());
    auto tilingDevice = static_cast<uint8_t *>(tiling_gpu_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t *>(softmaxlse.data_ptr());
    // =========================================================================
    // kernel 启动：根据 4 个维度组合选择模板实例化
    // =========================================================================
    // SplitFuse::FAInfer 模板参数说明：
    //   <ElementQ, ElementKV, ElementS, PAGED_CACHE_FLAG, MASK_TYPE, INPUT_LAYOUT, LSE_MODE>
    //
    //   ElementQ/KV  : 输入数据类型 — bfloat16_t(BF16) 或 half(FP16)
    //   ElementS     : softmax 中间计算类型 — float(FP32) 保证数值精度
    //   PAGED_CACHE  : bool — true=分页KV-Cache模式, false=标准模式
    //   MASK_TYPE    : MaskType 枚举 — MASK_CAUSAL(因果掩码) 或 NO_MASK(无掩码)
    //   INPUT_LAYOUT : inputLayout 枚举 — TND(变长布局) 或 BSND(标准batch布局)
    //   LSE_MODE     : LseModeT::OUT_ONLY — 仅输出 logsumexp，不做多 split 合并
    //
    // 分支组合共 2×2×2×2 = 16 种（dtype × paged × causal × varlen）
    // <<<blockDim, nullptr, aclStream>>> 为 AscendC kernel 启动语法：
    //   blockDim = AI Core 数量（多核并行），nullptr=无动态共享内存，aclStream=NPU流
    // =========================================================================
    if (is_bf16) {
        // ==================== BF16 数据类型 ====================
        if (paged_KV) {
            // ---- 分页 KV-Cache 模式 ----
            if (is_causal) {
                if (is_varlen_q) {
                    // BF16 + 分页 + 因果掩码 + 变长(TND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // BF16 + 分页 + 因果掩码 + 标准(BSND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    // BF16 + 分页 + 无掩码 + 变长(TND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // BF16 + 分页 + 无掩码 + 标准(BSND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            // ---- 非分页（标准）模式 ----
            if (is_causal) {
                if (is_varlen_q) { 
                    // BF16 + 标准 + 因果掩码 + 变长(TND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // BF16 + 标准 + 因果掩码 + 标准(BSND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    // BF16 + 标准 + 无掩码 + 变长(TND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // BF16 + 标准 + 无掩码 + 标准(BSND)
                    SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    } else {
        // ==================== FP16 数据类型 ====================
        if (paged_KV) {
            // ---- 分页 KV-Cache 模式 ----
            if (is_causal) {
                if (is_varlen_q) { 
                    // FP16 + 分页 + 因果掩码 + 变长(TND)
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // FP16 + 分页 + 因果掩码 + 标准(BSND)
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    // FP16 + 分页 + 无掩码 + 变长(TND)
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // FP16 + 分页 + 无掩码 + 标准(BSND)
                    SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        } else {
            // ---- 非分页（标准）模式 ----
            if (is_causal) {
                if (is_varlen_q) { 
                    // FP16 + 标准 + 因果掩码 + 变长(TND)
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // FP16 + 标准 + 因果掩码 + 标准(BSND)
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            } else {
                if (is_varlen_q) { 
                    // FP16 + 标准 + 无掩码 + 变长(TND)
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                } else {
                    // FP16 + 标准 + 无掩码 + 标准(BSND)
                    SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseModeT::OUT_ONLY><<<blockDim, nullptr, aclStream>>>(
                            fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice,
                            qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
                }
            }
        }
    }
    // ---- 返回结果：输出张量 out 和 logsumexp（softmaxlse） ----
    // out_accum 和 softmax_lse_accum 当前未使用，预留多 split 合并场景
    return {out, softmaxlse, out_accum, softmax_lse_accum};
}

// =============================================================================
// PYBIND11_MODULE — Python 模块注册
// =============================================================================
// 将 C++ 函数 mha_fwd 注册为 Python 模块 flash_attn_npu_3 的方法 "fwd"。
// 编译后生成 flash_attn_npu_3 扩展模块，Python 端可通过以下方式调用：
//   import flash_attn_npu_3
//   out, lse, _, _ = flash_attn_npu_3.fwd(q, k, v, ...)
//
// 模块名 flash_attn_npu_3 对应 setup.py 中 v3 的扩展模块名，
// Python 包 flash_attn_npu_v3 通过 torch.ops 或直接 import 来调用此 C++ 扩展。
// =============================================================================
PYBIND11_MODULE(flash_attn_npu_3, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass, with KV-cache");
}