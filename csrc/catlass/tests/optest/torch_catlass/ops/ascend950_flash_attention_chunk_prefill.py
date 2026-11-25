# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import torch
from torch import Tensor


def ascend950_flash_attention_chunk_prefill(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    actual_seq_lengths: Tensor,
    actual_seq_lengths_kv: Tensor,
    atten_mask: Tensor,
    block_table: Tensor,
    input_layout: str = "TND",
    num_heads: int = 0,
    num_key_value_heads: int = 0,
    block_size: int = 128,
    num_blocks: int = 2048,
    cache_layout: str = "nd",
    sparse_mode: int = 1,
) -> Tensor:
    """Run CATLASS flash attention inference on NPU tensors.

    Source: example 70_ascend950_ascend950_flash_attention_chunk_prefill.

    Args:
        query: Query tensor in TND layout ``(total_q_tokens, num_heads, head_dim)``.
        key: Key cache tensor. For paged KV cache, shape is
            ``(num_blocks, block_size, kv_heads, head_dim)``.
        value: Value cache tensor with the same layout as ``key``.
        actual_seq_lengths: Per-batch Q sequence lengths, shape ``(batch,)``.
        actual_seq_lengths_kv: Per-batch KV sequence lengths, shape ``(batch,)``.
        atten_mask: Attention mask tensor. Required when ``sparse_mode == 1``.
        block_table: Paged KV block table, shape ``(batch, max_num_blocks)``.
            Pass an empty tensor when paged cache is not used.
        input_layout: Only ``TND`` is supported.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.

    Returns:
        Output tensor with shape ``(total_q_tokens, num_heads, head_dim)``.
    """
    return torch.ops.catlass.ascend950_flash_attention_chunk_prefill(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        atten_mask,
        block_table,
        input_layout,
        num_heads,
        num_key_value_heads,
        block_size,
        num_blocks,
        cache_layout,
        sparse_mode
    )
