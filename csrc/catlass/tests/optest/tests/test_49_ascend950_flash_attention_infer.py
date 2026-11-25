# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import re

import pytest
import torch
import torch_npu
import torch_catlass


def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 49_ascend950_flash_attention_infer requires Ascend 950 NPU",
)


def _group_matmul(head: int, kv_head: int, left, right):
    group_num = head // kv_head
    chunks = []
    for i in range(kv_head):
        group_score = torch.matmul(
            left[i * group_num : (i + 1) * group_num].float(),
            right[i : i + 1].float(),
        )
        chunks.append(group_score)
    return torch.cat(chunks, dim=0)


def _ref_masked_attention(query, key, value, scale: float, mask):
    query_t = query.transpose(0, 1)
    key_t = key.transpose(0, 1).transpose(1, 2)
    value_t = value.transpose(0, 1)

    sim = _group_matmul(query_t.shape[0], key_t.shape[0], query_t, key_t) * scale
    if mask is not None:
        sim = sim + mask[: sim.shape[-2], : sim.shape[-1]].float() * (-3e38)

    attn = torch.softmax(sim, dim=-1)
    out = _group_matmul(query_t.shape[0], value_t.shape[0], attn.to(query.dtype), value_t)
    return out.transpose(0, 1)


def _reference_flash_attention(
    query,
    key_cache,
    value_cache,
    q_seqlen_list,
    kv_seqlen_list,
    block_table,
    mask,
    num_heads,
    kv_heads,
    head_dim,
    block_size,
):
    batch = len(q_seqlen_list)
    outputs = []
    cu_q = 0
    scale = 1.0 / (head_dim**0.5)

    for i in range(batch):
        q_len = int(q_seqlen_list[i])
        kv_len = int(kv_seqlen_list[i])
        q = query[cu_q : cu_q + q_len]

        keys = []
        values = []
        table = block_table[i]
        for j in range(kv_len):
            block_number = int(table[j // block_size].item())
            block_offset = j % block_size
            keys.append(key_cache[block_number, block_offset])
            values.append(value_cache[block_number, block_offset])
        key = torch.stack(keys, dim=0)
        value = torch.stack(values, dim=0)

        batch_mask = None
        if mask is not None:
            batch_mask = mask[cu_q : cu_q + q_len, :kv_len]

        outputs.append(_ref_masked_attention(q, key, value, scale, batch_mask))
        cu_q += q_len

    return torch.cat(outputs, dim=0).to(query.dtype)


def test_ascend950_flash_attention_infer_paged():
    """Compare Ascend950 flash attention infer against a PyTorch reference implementation."""
    torch.manual_seed(1)
    batch = 2
    q_seqlen = 64
    kv_seqlen = 128
    num_heads = 8
    kv_heads = 1
    head_dim = 128
    block_size = 128
    sparse_mode = 1

    q_seqlen_list = [q_seqlen] * batch
    kv_seqlen_list = [kv_seqlen] * batch
    num_tokens = sum(q_seqlen_list)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    max_num_blocks_per_seq = (max_kv_seqlen + block_size - 1) // block_size

    query = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="npu")
    key_cache = torch.randn(
        num_blocks, block_size, kv_heads, head_dim, dtype=torch.float16, device="npu"
    )
    value_cache = torch.randn(
        num_blocks, block_size, kv_heads, head_dim, dtype=torch.float16, device="npu"
    )

    block_table = [
        [max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
        for i in range(batch)
    ]
    block_table = torch.tensor(block_table, dtype=torch.int32, device="npu")

    actual_seq_lengths = torch.tensor(q_seqlen_list, dtype=torch.int64, device="npu")
    actual_seq_lengths_kv = torch.tensor(kv_seqlen_list, dtype=torch.int64, device="npu")

    # Dense uint8 mask [num_tokens, max_kv_seqlen] — same as example 49 mask.bin.
    ref_mask = torch.zeros(num_tokens, max_kv_seqlen, dtype=torch.uint8, device="cpu")
    cu_q = 0
    for q_len, k_len in zip(q_seqlen_list, kv_seqlen_list):
        tri = torch.triu(torch.ones(q_len, q_len), diagonal=1).to(torch.uint8)
        ref_mask[cu_q : cu_q + q_len, 0:q_len] = tri
        cu_q += q_len

    atten_mask = ref_mask.to(device="npu")

    result = torch_catlass.ascend950_flash_attention_infer(
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
        sparse_mode,
    )

    expected = _reference_flash_attention(
        query.cpu(),
        key_cache.cpu(),
        value_cache.cpu(),
        q_seqlen_list,
        kv_seqlen_list,
        block_table.cpu(),
        ref_mask.float(),
        num_heads,
        kv_heads,
        head_dim,
        block_size,
    )

    assert result.shape == (num_tokens, num_heads, head_dim)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result.cpu().float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.cpu().float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
