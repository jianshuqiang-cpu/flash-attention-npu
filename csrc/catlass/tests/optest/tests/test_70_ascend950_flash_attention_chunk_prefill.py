# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import numpy as np
import pytest
import torch

import torch_catlass
from common import only_on_3510


def group_matmul(head, kv_head, left, right):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(
            left[i * group_num : (i + 1) * group_num, :, :].astype(np.float32),
            right[i : (i + 1), :, :].astype(np.float32),
        )
        score = group_score if score is None else np.concatenate((score, group_score), 0)
    return score


def softmax_numpy(sim):
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim_sub = sim - row_max
    sim_sub = np.exp(sim_sub)
    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
    soft_res = sim_sub / row_sum
    lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)
    return soft_res, lse, row_max


def softmax1(qk_result, is_first, gm, is_kvs_last_loop, data_type=np.float16):
    sim = qk_result
    lm = np.max(sim, axis=-1, keepdims=True)
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = np.maximum(gm, lm)
        dm = gm - hm

    gm = hm
    sim_sub = sim - hm
    sim_sub = np.exp(sim_sub.astype(np.float32))

    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

    return sim_sub, row_sum, dm, gm


def ref_masked_attention(query, key, value, scale: float, mask):
    query = query.numpy()
    mask = mask.numpy()
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 2, 0))
    sim_high = group_matmul(query.shape[0], key.shape[0], query, key)
    sim_low_prec = sim_high.astype(np.float16) * np.float16(scale)
    sim_high = sim_high * scale
    if mask is not None:
        sim_high = sim_high + (mask[: sim_high.shape[-2], : sim_high.shape[-1]]).astype(np.float32)
        sim_low_prec = sim_low_prec + (mask[: sim_high.shape[-2], : sim_high.shape[-1]]).astype(
            np.float16
        )

    p_high, lse_high, gm = softmax_numpy(sim_high)
    p_low_prec, lse_low_prec, gm_low_prec = softmax_numpy(sim_low_prec)
    lse = lse_high.astype(query.dtype)
    lse_high = lse_high.astype(np.float32)
    p = p_high.astype(query.dtype)
    p_high = p_high.astype(np.float32)
    value = np.transpose(value, (1, 0, 2))

    out_low_prec = group_matmul(query.shape[0], key.shape[0], p_low_prec, value)
    out_high = group_matmul(query.shape[0], key.shape[0], p_high, value)
    out = group_matmul(query.shape[0], key.shape[0], p, value)
    out_low_prec = np.transpose(out_low_prec, (1, 0, 2))
    out_high = np.transpose(out_high, (1, 0, 2))
    out = np.transpose(out, (1, 0, 2))
    out_low_prec = out_low_prec.astype(np.float16)
    out = out.astype(query.dtype)
    return out, out_high, out_low_prec, lse, lse_high, gm


def ref_single_query_cached_kv_attention(
    query,
    key_cache,
    value_cache,
    q_seqlen_list,
    k_seqlen_list,
    block_tables,
    global_mask,
    num_heads,
    kv_heads,
    qk_head_size,
    v_head_size,
    block_size,
):
    num_heads = num_heads
    kv_heads = kv_heads
    head_size_qk = qk_head_size
    head_size_vo = v_head_size
    block_size = block_size

    batch = len(q_seqlen_list)
    cu_seqlen = 0
    kv_seqlen_now = 0
    output = []
    for i in range(batch):
        q_seqlen = int(q_seqlen_list[i])
        k_seqlen = int(k_seqlen_list[i])
        print(f"batch:{i}")
        print(f"q_seqlen:{q_seqlen}")
        print(f"k_seqlen:{k_seqlen}")
        q = query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
        keys = []
        values = []
        block_table = block_tables[i]

        for j in range(k_seqlen):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, block_offset, :]
            k = k.reshape(kv_heads, head_size_qk)
            keys.append(k)
            v = value_cache[block_number, :, block_offset, :]
            v = v.reshape(kv_heads, head_size_vo)
            values.append(v)
        keys = np.stack(keys, axis=0)
        values = np.stack(values, axis=0)

        scale = 1.0 / (head_size_qk**0.5)

        mask = global_mask[cu_seqlen : (cu_seqlen + q_seqlen), :]

        out_normal, _, out_low_prec, lse, _, gm = ref_masked_attention(q, keys, values, scale, mask)

        out = out_normal
        out = out.reshape(-1, num_heads, head_size_vo)

        output.append(torch.tensor(out))

        cu_seqlen += q_seqlen
        kv_seqlen_now += k_seqlen
    return torch.cat(output, dim=0).to(query.dtype)


@only_on_3510
def test_ascend950_ascend950_flash_attention_chunk_prefill_paged_mask():
    """Compare ascend950 flash attention chunkprefill against a PyTorch reference implementation."""
    torch.manual_seed(1)
    batch = 2
    q_seqlen = 64
    kv_seqlen = 128
    num_heads = 8
    kv_heads = 1
    qk_head_dim = 128
    v_head_dim = 128
    block_size = 128
    sparse_mode = 1
    cache_layout = "nd"

    q_seqlen_list = [q_seqlen] * batch
    q_seqlen_list_npu = [i * q_seqlen for i in range(batch + 1)]
    kv_seqlen_list = [kv_seqlen] * batch
    num_tokens = sum(q_seqlen_list)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    max_num_blocks_per_seq = (max_kv_seqlen + block_size - 1) // block_size

    query = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=torch.float16, device="npu")
    key_cache = torch.randn(
        num_blocks, kv_heads, block_size, qk_head_dim, dtype=torch.float16, device="npu"
    )
    value_cache = torch.randn(
        num_blocks, kv_heads, block_size, v_head_dim, dtype=torch.float16, device="npu"
    )

    block_table = []
    for i in range(batch):
        block_table.append([max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)])
    block_table = torch.tensor(block_table, dtype=torch.int32, device="npu")

    actual_seq_lengths = torch.tensor(q_seqlen_list_npu, dtype=torch.int64, device="npu")
    actual_seq_lengths_kv = torch.tensor(kv_seqlen_list, dtype=torch.int64, device="npu")

    mask = torch.zeros(num_tokens, max_kv_seqlen, dtype=torch.float16, device="npu")
    pre_mask_factor = -65500
    cu_q = 0
    for i in range(batch):
        q_len = q_seqlen_list[i]
        k_len = kv_seqlen_list[i]
        tri = torch.triu(torch.ones(q_len, q_len, device="npu"), diagonal=1)
        tri *= pre_mask_factor
        mask[cu_q : cu_q + q_len, k_len - q_len : k_len] = tri
        cu_q += q_len

    atten_mask = torch.triu(torch.ones(2048, 2048, device="npu"), diagonal=1).to(torch.int8)

    result = torch_catlass.ascend950_flash_attention_chunk_prefill(
        query,
        key_cache,
        value_cache,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        atten_mask,
        block_table,
        "TND",
        num_heads,
        kv_heads,
        block_size,
        num_blocks,
        cache_layout,
        sparse_mode,
    )

    expected = ref_single_query_cached_kv_attention(
        query.cpu(),
        key_cache.cpu(),
        value_cache.cpu(),
        q_seqlen_list,
        kv_seqlen_list,
        block_table.cpu(),
        mask.cpu(),
        num_heads,
        kv_heads,
        qk_head_dim,
        v_head_dim,
        block_size,
    )

    assert result.shape == (num_tokens, num_heads, qk_head_dim)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert expected.dtype == torch.float16

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result.cpu().float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.cpu().float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
