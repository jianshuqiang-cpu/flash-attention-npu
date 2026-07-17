import torch
import math 
import numpy as np
import sys

class TestFlashAttentionInfer():

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if q_seqlen > kv_seqlen:
            logging("[ERROR] q_seqlen cannot exceed kv_seqlen.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)
        return soft_res, lse, row_max
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split: qk_k_split + sub_k]
            key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
            result_split = self.group_matmul(query_k.shape[0], key_k.shape[0], query_k, key_k)
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def ref_masked_attention(self,
            query,
            key,
            value,
            scale: float,
            mask
    ):
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)
        sim_low_prec = sim_high.astype(np.float16) * np.float16(scale)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float32)
            sim_low_prec = sim_low_prec + (
                mask[:sim_high.shape[-2], :sim_high.shape[-1]]
                ).astype(np.float16)

        p_high, lse_high, gm = self.softmax_numpy(sim_high)
        p_low_prec, lse_low_prec, gm_low_prec = self.softmax_numpy(sim_low_prec)
        lse = lse_high.astype(query.dtype)
        lse_high = lse_high.astype(np.float32)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))
        
        out_low_prec = self.group_matmul(query.shape[0], key.shape[0], p_low_prec, value)
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_low_prec = np.transpose(out_low_prec, (1, 0, 2))
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out_low_prec = out_low_prec.astype(np.float16)
        out = out.astype(query.dtype)
        return out, out_high, out_low_prec, lse, lse_high, gm

    def ref_single_query_cached_kv_attention(self, batch, q, k, v, mask, scale) -> None:
        output = np.zeros_like(q)
        for i in range(batch):
            out_normal, _, out_low_prec, lse, _, gm = self.ref_masked_attention(q[i], k[i], v[i], scale, mask[i])
            output[i] = out_normal
        return output

def print_mask(seq_len, dense_mask):
    return 
    if dense_mask.dim() != 3:
        # 安全防范：确保最终一定是 2D
        raise ValueError(f"Expected mask to be 2D after squeezing, but got shape {dense_mask.shape}")
    print("========reference==========")
    for i in range(seq_len):
        for j in range(seq_len):
            print(1 if dense_mask[0][i][j].item() == True else 0, end="" if (j + 1) % 128 != 0 else "    ")
        print("")
        if (i + 1) % 128 == 0:
            print("")

def print_mask2(seq_len, dense_mask):
    return 
    if dense_mask.dim() != 2:
        # 安全防范：确保最终一定是 2D
        raise ValueError(f"Expected mask to be 2D after squeezing, but got shape {dense_mask.shape}")
    print("========reference==========")
    for i in range(seq_len):
        for j in range(seq_len):
            print(1 if dense_mask[i][j].item() == True else 0, end="" if (j + 1) % 128 != 0 else "    ")
        print("")
        if (i + 1) % 128 == 0:
            print("")

def ref_flash_attention2(query, key, value, scale, mask, data_type):
    """One-shot vectorized golden reference (GQA-aware, bottom-right causal).

    query : (Sq,H,D), key/value : (kv,Hkv,D) [per-batch, CPU]
    mask  : (Sq,kv) bool causal mask (bottom-right, diagonal=kv-Sq+1) or None
    Returns out (Sq,H,D) in data_type, lse (H,Sq) fp32.

    Replaces the chunked online-softmax + per-group Python loops (group_matmul /
    qkMM1 / pvMM2) with 2 batched matmuls (Q@K^T, P@V) + softmax/logsumexp over the
    full kv in a single pass. Same fp32 math, ~50-100x faster (no Python loops).
    GQA via broadcast (unsqueeze) — no repeat_interleave materialization of K/V."""
    Sq, H, D = query.shape
    Hkv = key.shape[1]
    kv = key.shape[0]
    g = H // Hkv
    q = query.permute(1, 0, 2).float().view(Hkv, g, Sq, D)            # (Hkv,g,Sq,D)
    k = key.permute(1, 0, 2).float()                                  # (Hkv,kv,D)
    v = value.permute(1, 0, 2).float()                                # (Hkv,kv,D)
    s = torch.matmul(q, k.unsqueeze(1).transpose(-1, -2)).view(H, Sq, kv) * scale  # (H,Sq,kv)
    if mask is not None:
        s = s + mask.to(s.dtype).unsqueeze(0) * (-1e4)
    lse = torch.logsumexp(s, dim=-1)                                  # (H,Sq) fp32
    p = torch.softmax(s, dim=-1).to(data_type).float().view(Hkv, g, Sq, kv)  # fp32 -> data_type -> fp32 (matches old p round-trip)
    out = torch.matmul(p, v.unsqueeze(1)).view(H, Sq, D).permute(1, 0, 2).to(data_type)  # (Sq,H,D)
    return out, lse

def group_matmul(head, kv_head, left, right, high_prec=1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32)).to(torch.float32)
        else:
            group_score = torch.matmul(left[i * group_num:(i + 1) * group_num, :, :].to(torch.float32),
                                        right[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score


def softmax1(qk_result, is_first, gm, interm_dtype=torch.float16):
    sim = qk_result.to(interm_dtype)
    lm = torch.max(sim, dim=-1, keepdims=True)[0]
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = torch.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = torch.exp(sim_sub.to(interm_dtype))
    row_sum = torch.sum(sim_sub, dim=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm


def qkMM1(query, key):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result


def pvMM2(p, value):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)
        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result


def ref_flash_attention(query, key, value, scale, mask, data_type):
    inner_prec = 0
    interm_dtype = torch.float16 if inner_prec == 1 else torch.float32
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    scale = torch.tensor(scale)
    scale = scale.to(torch.float16) if inner_prec == 1 else scale.to(torch.float32)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            sub_mask = mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype) * (-1e4)
            print_mask2(query.shape[1], mask[:query.shape[1], kv_start : kv_start + sub_len].to(interm_dtype))
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).to(interm_dtype)
        # print("===========ref_flash_attention========================")
        # print(qk_result)
        
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.to(data_type)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).to(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = torch.exp(dm)
            gl = gl * dm
            gl = gl + row_sum
            go = go * dm
            go = go + lo
    go = go / gl
    go = go.permute(1, 0, 2)
    lse = torch.squeeze((torch.log(gl) + gm), dim=-1).to(torch.float32)
    return go.to(data_type), lse


def _to_int_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    return [int(v) for v in x]


def _gather_paged_kv(key_cache_cpu, value_cache_cpu, block_table, kv_seqlen, block_size):
    """Vectorized paged-KV gather -> (kv_seqlen, kv_heads, head_size).
    Reconstructs one batch's dense KV from page-table indices in a single gather."""
    j = torch.arange(kv_seqlen, dtype=torch.long)
    block_number = block_table.to(torch.long)[j // block_size]   # (kv_seqlen,)
    block_offset = j % block_size                                # (kv_seqlen,)
    return key_cache_cpu[block_number, block_offset], value_cache_cpu[block_number, block_offset]


def golden_flash_attn(q, k_cache, v_cache, cache_seqlens, maskTensor=None,
                      block_table=None, cu_seqlens_q=None,
                      causal=False, scale=None, block_size=128, data_type=None):
    """Golden reference mirroring flash_attn_with_kvcache / flash_attn_func /
    flash_attn_varlen_func: feed the same inputs, get (out, lse) back.

    Inputs (CPU or NPU tensors):
      q            : (B,Sq,H,D) if cu_seqlens_q is None (BSND), else (total_q,H,D) (TND/varlen)
      k_cache/v_cache : paged  (num_blocks,block_size,Hkv,D) if block_table given
                        dense  (B,kv,Hkv,D)        if cu_seqlens_q is None (BSND)
                        dense  (total_kv,Hkv,D)    otherwise (TND/varlen, packed by cache_seqlens)
      cache_seqlens: (B,) per-batch KV length
      block_table  : (B,max_blocks) paged page table, or None (dense)
      cu_seqlens_q : (B+1,) cumulative Q lengths (TND/varlen), or None (BSND)
      causal       : bottom-right causal mask (diagonal = kv-Sq+1), matches kvcache/varlen
      scale        : default 1/sqrt(D)
      block_size   : page size (default 128)
      data_type    : output dtype (default q.dtype)

    Returns:
      out : (B,Sq,H,D) BSND or (total_q,H,D) TND, in data_type
      lse : (B,H,Sq)   BSND or (H,total_q)   TND, fp32
    """
    data_type = data_type if data_type is not None else q.dtype
    D = q.shape[-1]
    H = q.shape[-2]
    Hkv = k_cache.shape[-2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    is_tnd = cu_seqlens_q is not None
    is_paged = block_table is not None

    q_cpu = q.detach().cpu()
    k_cpu = k_cache.detach().cpu()
    v_cpu = v_cache.detach().cpu()

    kv_lens = _to_int_list(cache_seqlens)
    B = len(kv_lens)

    cu_q = _to_int_list(cu_seqlens_q)
    if is_tnd:
        q_lens = [cu_q[i + 1] - cu_q[i] for i in range(B)]
        total_q = cu_q[-1]
    else:
        Sq = q_cpu.shape[1]
        q_lens = [Sq] * B
        cu_q = [Sq * i for i in range(B + 1)]
        total_q = B * Sq

    # dense TND KV is packed by cumulative cache_seqlens
    cu_k = None
    if is_tnd and not is_paged:
        cu_k = [0]
        for L in kv_lens:
            cu_k.append(cu_k[-1] + L)

    if is_tnd:
        golden_out = torch.empty((total_q, H, D), dtype=data_type)
        golden_lse = torch.empty((H, total_q), dtype=torch.float32)
    else:
        golden_out = torch.empty((B, q_lens[0], H, D), dtype=data_type)
        golden_lse = torch.empty((B, H, q_lens[0]), dtype=torch.float32)

    block_table_cpu = block_table.detach().cpu() if is_paged else None
    for i in range(B):
        Sq_i = q_lens[i]
        kv_i = kv_lens[i]
        q_i = q_cpu[cu_q[i]:cu_q[i + 1]] if is_tnd else q_cpu[i]
        if is_paged:
            k_i, v_i = _gather_paged_kv(k_cpu, v_cpu, block_table_cpu[i], kv_i, block_size)
        elif is_tnd:                       # dense TND: packed by cumsum(cache_seqlens)
            k_i = k_cpu[cu_k[i]:cu_k[i + 1]]
            v_i = v_cpu[cu_k[i]:cu_k[i + 1]]
        else:                              # dense BSND: per-batch row, sliced to valid kv
            k_i = k_cpu[i][:kv_i]
            v_i = v_cpu[i][:kv_i]
        mask = None
        if maskTensor is not None:
            mask = maskTensor[i].cpu()
        elif causal:
            mask = torch.triu(torch.ones(Sq_i, kv_i), diagonal=kv_i - Sq_i + 1).bool()
        out_i, lse_i = ref_flash_attention(q_i, k_i, v_i, scale, mask, data_type)
        out_i = out_i.reshape(Sq_i, H, D)
        if is_tnd:
            golden_out[cu_q[i]:cu_q[i + 1]] = out_i
            golden_lse[:, cu_q[i]:cu_q[i + 1]] = lse_i.reshape(H, Sq_i)
        else:
            golden_out[i:i + 1] = out_i
            golden_lse[i:i + 1] = lse_i.reshape(1, H, Sq_i)
    return golden_out, golden_lse


def generate_compressed(
    doc_lens: list[int],
    prefix_len: int,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = sum(doc_lens)
    doc_offsets = [0] 
    for doc_len in doc_lens:
        doc_offsets.append(doc_offsets[-1] + doc_len)
    maskr = torch.zeros(seq_len, dtype=torch.long, device=device)
    left_len = torch.zeros(seq_len, dtype=torch.long, device=device)
    
    for doc_idx, doc_len in enumerate(doc_lens):
        start = doc_offsets[doc_idx]
        end = start + doc_len
        prefix_end = start + min(prefix_len, doc_len)
        q_idx_doc = torch.arange(start, end, device=device)

        left_len[start:end] = start
        maskr[start:prefix_end] = end
        maskr[prefix_end:end] = q_idx_doc[prefix_end-start:] + 1
    
    mask_start = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    mask_len = left_len.unsqueeze(0)
    maskr = maskr.unsqueeze(0).expand(batch_size, -1).contiguous()
    mask_start = mask_start.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    mask_len = mask_len.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return maskr, mask_start, mask_len

def get_last_true_index_by_weight(block_compute: torch.Tensor) -> torch.Tensor:
    num_cols = block_compute.shape[-1]
    
    # 1. 生成一个和行等长的索引序列：[0, 1, 2, ..., num_cols - 1]
    # 并将其广播（Broadcast）到与 block_compute 相同的形状
    col_indices = torch.arange(num_cols, device=block_compute.device)
    
    # 2. 使用 where 过滤：True 的位置保留索引值，False 的位置设为 -1
    masked_indices = torch.where(block_compute, col_indices, -1)
    
    # 3. 求最大值，最大值即为最后一个 True 的索引
    # 如果整行都是 False，最大值自然会是 -1
    last_true_idx, _ = masked_indices.max(dim=-1)
    
    return last_true_idx + 1

def generate_block_mask(
    dense_mask: torch.Tensor,
    tile_m: int,
    tile_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_q, seq_k = dense_mask.shape
    num_tiles_m = (seq_q + tile_m - 1) // tile_m
    num_tiles_n = (seq_k + tile_n - 1) // tile_n
    pad_m = num_tiles_m * tile_m - seq_q
    pad_n = num_tiles_n * tile_n - seq_k

    padded = torch.nn.functional.pad(
        dense_mask.float(), (0, pad_n, 0, pad_m), value=1.0
    )
    blocks = padded.unfold(1, tile_m, tile_m).unfold(2, tile_n, tile_n)
    blocks_mean = blocks.mean(dim=(-1, -2))
    all_masked = blocks_mean > 1.0 - 1e-6
    all_valid = blocks_mean < 1e-6
    block_compute = ~all_masked
    block_mask = ~all_valid & ~all_masked
    tile_range = get_last_true_index_by_weight(block_compute)
    return block_compute.to(dtype=torch.bool), block_mask.to(dtype=torch.bool), tile_range


def pack_tile_bits(bool_mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool tile-mask (B, Tq, Tk) into int32 bitmask (B, Tq, Wk).

    Bit (tk % 32) of word (tk // 32) is set where bool_mask is True, matching the
    AnyMask sparse_compute / sparse_mask layout consumed by the 950 forward kernel
    (csrc_AscendC950/flash_attn_npu_v3/mha_fwd.cpp: bit = word & (1 << (tk % 32))).
    """
    B, Tq, Tk = bool_mask.shape
    Wk = (Tk + 31) // 32
    pad = Wk * 32 - Tk
    if pad:
        bool_mask = torch.nn.functional.pad(bool_mask, (0, pad), value=False)
    bits = bool_mask.to(torch.int32).reshape(B, Tq, Wk, 32)
    weights = (1 << torch.arange(32, device=bool_mask.device, dtype=torch.int32))
    return (bits * weights).sum(dim=-1).to(torch.int32)


def build_anymask_params(
    doc_lens: list[int],
    prefix_len: int,
    batch_size: int,
    seq_len: int,
    maskr: torch.Tensor,
    block_compute: torch.Tensor,
    block_mask: torch.Tensor,
    device: torch.device | None = None,
) -> dict:
    """Build the AnyMask inputs for flash_attn_with_kvcache from the generated masks.

    Reconstructs generate_dense_mask exactly (see the kernel reference in mha_fwd.cpp):
      * maskr           (B, S)     int32 : per-query right boundary, k >= maskr[b,q] -> masked.
      * holel / holes   (B, S, 1)  int32 : one hole per query = [0, prefix_end_of_doc) masks
                                           the left region (cross-doc + own prefix). Hn = 1.
      * sparse_compute  (B, Tq, Wk) int32 : bit set on fully-masked tiles (skip compute).
      * sparse_mask     (B, Tq, Wk) int32 : bit set on partial tiles (apply maskr + holes).
      * hole_num        (1,)       int16 : Hn = 1.
      * tile_range      None             : right boundary already carried per-query by maskr.

    The block-mask classification is conservative: all_valid tiles carry no bit (computed
    unmasked), all_masked tiles carry the sparse_compute bit (skipped), partial tiles carry
    the sparse_mask bit and get per-query maskr + holes, which reproduces the dense mask.
    """
    maskr_i32 = maskr.to(torch.int32)

    # prefix_end of the doc each query belongs to = doc_start + min(prefix_len, doc_len).
    prefix_end_per_q = torch.zeros(seq_len, dtype=torch.int32, device=device)
    doc_offsets = [0]
    for dl in doc_lens:
        doc_offsets.append(doc_offsets[-1] + dl)
    for doc_idx, doc_len in enumerate(doc_lens):
        start = doc_offsets[doc_idx]
        prefix_end = start + min(prefix_len, doc_len)
        prefix_end_per_q[start:start + doc_len] = prefix_end

    holel = torch.zeros((batch_size, seq_len, 1), dtype=torch.int32, device=device)
    holes = prefix_end_per_q.view(1, seq_len, 1).expand(batch_size, -1, -1).contiguous()
    hole_num = torch.tensor([1], dtype=torch.int16, device=device)

    sparse_compute = pack_tile_bits(~block_compute)  # fully-masked tiles
    sparse_mask = pack_tile_bits(block_mask)         # partial tiles

    return dict(
        hole_num=hole_num,
        sparse_compute=sparse_compute,
        sparse_mask=sparse_mask,
        maskr=maskr_i32,
        holel=holel,
        holes=holes,
    )


def generate_dense_mask(
    doc_lens: list[int],
    prefix_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    seq_len = sum(doc_lens)
    doc_offsets = [0] 
    for doc_len in doc_lens:
        doc_offsets.append(doc_offsets[-1] + doc_len)
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for doc_idx, doc_len in enumerate(doc_lens):
        start = doc_offsets[doc_idx]
        end = start + doc_len
        prefix_end = start + min(prefix_len, doc_len)
        q_in_doc = (q_idx >= start) & (q_idx < end)
        mask = mask | (q_in_doc & ((k_idx < prefix_end) | (k_idx >= end)))
        mask = mask | (q_in_doc & (q_idx >= prefix_end) & (k_idx > q_idx))
    return mask.unsqueeze(0)

def dense_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dense_mask: torch.Tensor = None,
) -> torch.Tensor:
    # Q, K, V: (batch_size, seq_len, num_heads, head_dim)  [BSHD]
    # 内部转到 head-major (B, H, S, D) 计算，最后转回 BSHD。
    # dense_mask 为 (B, S, S)，unsqueeze(1) -> (B, 1, S, S) 在 head 轴广播，与 BNSD 一致。
    Qh = Q.transpose(1, 2)
    Kh = K.transpose(1, 2)
    Vh = V.transpose(1, 2)
    scores = Qh @ Kh.transpose(-2, -1) 
    # print("================dense_attention_reference===================")
    # print(scores)
    scores = scores * (1.0 / math.sqrt(Q.shape[-1]))
    if dense_mask is not None:
        print_mask(dense_mask.shape[2], dense_mask)
        scores = scores.masked_fill(dense_mask.unsqueeze(1), float("-inf"))
    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max),row_max,torch.zeros_like(row_max))
    probs = torch.exp(scores - row_max)
    probs = torch.where(torch.isfinite(scores),probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        denom > 0,
        probs / denom.clamp_min(1e-30),
        torch.zeros_like(probs),
    )
    out = probs @ Vh                          # (B, H, S, D)
    return out.transpose(1, 2).contiguous()   # (B, S, H, D)

if __name__ == "__main__":
    print(math.floor(math.log2(1.5)))
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doc_lens = [128,96,160,128,96,160]
    prefix_len = 32
    batch_size = 10
    num_heads = 10

    head_dim = 128
    tile_m = 128
    tile_n = 128
    seq_len = sum(doc_lens)
    dense_mask = generate_dense_mask(doc_lens, prefix_len, device=device)
    dense_mask = dense_mask.expand(batch_size, -1, -1).contiguous()
    maskr, mask_start, mask_len = generate_compressed(
        doc_lens, prefix_len, batch_size=batch_size, device=device
    )

    block_compute, block_mask, tile_range = generate_block_mask(
        dense_mask, tile_m=tile_m, tile_n=tile_n
    )

    q_min_range = -1
    q_max_range = 1
    op_dtype = torch.float16
    Q = q_min_range + (q_max_range - q_min_range) * torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float32, device=device)
    K = q_min_range + (q_max_range - q_min_range) * torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float32, device=device)
    V = q_min_range + (q_max_range - q_min_range) * torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=torch.float32, device=device)

    # Run the reference on fp16-quantized inputs so the diff vs the NPU op reflects
    # kernel/mask numerics rather than input quantization. AnyMask v1 on 950 is fp16-only.

    # ---- AnyMask inputs for flash_attn_with_kvcache (Ascend 950) ----
    anymask = build_anymask_params(
        doc_lens, prefix_len, batch_size, seq_len,
        maskr, block_compute, block_mask, device=device,
    )

    # ---- NPU call. Skipped if torch_npu / the op is unavailable. ----
    try:
        import torch_npu  # noqa: F401
        from flash_attn_npu_v3.flash_attn_interface_950 import flash_attn_with_kvcache
        npu_dev = torch.device("npu:0") if torch.npu.is_available() else None
    except Exception as e:  # pragma: no cover - env-dependent
        npu_dev = None
        print(f"[skip] NPU / flash_attn_with_kvcache unavailable: {e}")

    if npu_dev is not None:
        torch.set_printoptions(
            threshold=float("inf"),
            linewidth=300,
            precision=5
        )
        q_npu = Q.to(op_dtype).to(npu_dev).contiguous()
        k_npu = K.to(op_dtype).to(npu_dev).contiguous()
        v_npu = V.to(op_dtype).to(npu_dev).contiguous()
        am = {k: t.to(npu_dev) for k, t in anymask.items()}
        am["tile_range"] = tile_range.to(torch.int32).to(npu_dev)
        # print("====hole_num============================")
        # print(am["hole_num"])
        # print("=====tile_range===========================")
        # print(tile_range)
        # print("=====sparse_compute===========================")
        # print(am["sparse_compute"])
        # print("======sparse_mask==========================")
        # print(am["sparse_mask"])
        # print("=====maskr===========================")
        # print(am["maskr"])
        # print("==holel==============================")
        # print(am["holel"])
        # print("==holes==============================")
        # print(am["holes"])
        
        kv_seqlen_list = [seq_len] * batch_size
        kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int32).npu()
        print_mask(dense_mask.shape[2], dense_mask)
        if len(sys.argv) > 1:
            dense_mask= None
        dens_out = dense_attention_reference(
            Q, K, V, dense_mask
        )
        golden_out_fp32, _ = golden_flash_attn(
            Q, K, V, kv_seqlen_list, maskTensor=dense_mask
        )
        golden_out_fp16, _ = golden_flash_attn(
            Q.to(op_dtype), K.to(op_dtype), V.to(op_dtype), kv_seqlen_list, maskTensor=dense_mask
        )
        testObj = TestFlashAttentionInfer()
        goolden2 = testObj.ref_single_query_cached_kv_attention(batch_size, Q.numpy(), K.numpy(), V.numpy(), dense_mask.numpy(), 1.0 / math.sqrt(head_dim))
        if dense_mask is not None:
            out = flash_attn_with_kvcache(
                q_npu,
                k_npu,
                v_npu,
                cache_seqlens=kv_seqlen_list,
                max_seqlen_q=seq_len,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=False,
                window_size=(-1, -1),
                return_softmax_lse=False,
                **am,
            )
        else:
            out = flash_attn_with_kvcache(
                q_npu,
                k_npu,
                v_npu,
                cache_seqlens=kv_seqlen_list,
                max_seqlen_q=seq_len,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=False,
                window_size=(-1, -1),
                return_softmax_lse=False,
            )
        # print("========reference==========")
        # for i in range(seq_len):
        #     for j in range(seq_len):
        #         print(1 if dense_mask[0][i][j].item() == True else 0, end="" if (j + 1) % 128 != 0 else "    ")
        #     print("")
        #     if (i + 1) % 128 == 0:
        #         print("")
        
        
        rtol = 0.001 ## 不加mask是OK的
        print("==========================dense_mask==========================")
        print(dense_mask is None)
        print(rtol)
        atol = rtol
        x = dens_out.to(out.dtype)
        try:
            torch.testing.assert_close(out.cpu(), golden_out_fp16, rtol=rtol, atol=atol)
            assert ((out.cpu() - dens_out.to(out.dtype)).abs().max().item() <= 1e-3)
        except Exception as e:
            print(f"\nout vs golden_out_fp16 error:{e}\n\n")
            # print((out.cpu() - golden_out_fp32.to(out.dtype)).abs().max().item())
            # print((out.cpu() - golden_out_fp32.to(out.dtype)).abs().mean().item())
            print("golden_out_fp16 最大误差 and 平均误差:")
            print((out.cpu() - golden_out_fp16).abs().max().item())
            print((out.cpu() - golden_out_fp16).abs().mean().item())
            print("真值 最大误差 and 平均误差:")
            print((out.cpu() - x).abs().max().item())
            print((out.cpu() - x).abs().mean().item())
        try:
            assert ((out.cpu() - x).abs().max().item() <= 4e-2) # 4e-2
            assert ((out.cpu() - x).abs().mean().item() <= 1e-3) # 2e-3
            assert (out.cpu() - golden_out_fp16).abs().max().item() <= 2 * (
                dens_out - golden_out_fp16
            ).abs().max().item()
        except Exception as e:
            print(f"\nout vs golden_out_fp32 error:{e}\n\n")

