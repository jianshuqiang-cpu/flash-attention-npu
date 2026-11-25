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


def ascend950_flash_attention_infer(
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
    sparse_mode: int = 0,
) -> Tensor:
    """Run CATLASS Ascend950 flash attention inference on NPU tensors.

    Source: example 49_ascend950_flash_attention_infer.

    Args:
        query: Query tensor in TND layout ``(total_q_tokens, num_heads, head_dim)``.
        key: Key cache tensor. For paged KV cache, shape is
            ``(num_blocks, block_size, kv_heads, head_dim)``.
        value: Value cache tensor with the same layout as ``key``.
        actual_seq_lengths: Per-batch Q sequence lengths, shape ``(batch,)``.
        actual_seq_lengths_kv: Per-batch KV sequence lengths, shape ``(batch,)``.
        atten_mask: Dense attention mask ``(total_q_tokens, max_kv_seqlen)`` in
            ``torch.uint8`` (0/1), matching example 49 ``mask.bin``. Required
            when ``sparse_mode == 1``.
        block_table: Paged KV block table, shape ``(batch, max_num_blocks)``.
            Pass an empty tensor when paged cache is not used.
        input_layout: Only ``TND`` is supported.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        sparse_mode: ``0`` for no mask, ``1`` for chunked causal mask.

    Returns:
        Output tensor with shape ``(total_q_tokens, num_heads, head_dim)``.
    """
    if sparse_mode == 1 and atten_mask.dtype != torch.uint8:
        atten_mask = atten_mask.to(torch.uint8)

    return torch.ops.catlass.ascend950_flash_attention_infer(
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
        sparse_mode,
    )
