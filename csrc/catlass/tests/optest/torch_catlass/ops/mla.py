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


def mla(
    query_nope: Tensor,
    query_rope: Tensor,
    key_cache: Tensor,
    key_rope_cache: Tensor,
    actual_seq_lengths: Tensor,
    actual_seq_lengths_kv: Tensor,
    block_table: Tensor,
    num_heads: int = 0,
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
) -> Tensor:
    """Run CATLASS MLA inference on NPU tensors.

    Source: example 19_mla.

    Args:
        query_nope: Query nope tensor, shape ``(total_q_tokens, num_heads, head_dim)``.
        query_rope: Query rope tensor, shape ``(total_q_tokens, num_heads, rope_dim)``.
        key_cache: Paged KV nope cache, shape ``(num_blocks, block_size, kv_heads, head_dim)``.
        key_rope_cache: Paged KV rope cache, shape ``(num_blocks, block_size, kv_heads, rope_dim)``.
        actual_seq_lengths: Per-batch Q sequence lengths (int32), shape ``(batch,)``.
        actual_seq_lengths_kv: Per-batch KV sequence lengths (int32), shape ``(batch,)``.
        block_table: Paged KV block table, shape ``(batch, max_num_blocks)``.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        sparse_mode: ``0`` for no mask, ``1`` for chunked causal mask.

    Returns:
        Output tensor with shape ``(total_q_tokens, num_heads, head_dim)``.
    """
    return torch.ops.catlass.mla(
        query_nope,
        query_rope,
        key_cache,
        key_rope_cache,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        num_heads,
        num_key_value_heads,
        sparse_mode,
    )
