# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


from typing import Optional

import torch
from torch import Tensor


def ascend950_fp8_mx_flash_attention_infer(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    actual_seq_lengths: Tensor,
    actual_seq_lengths_kv: Tensor,
    atten_mask: Tensor,
    block_table: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    p_scale: Optional[Tensor] = None,
    input_layout: str = "TND",
    num_heads: int = 0,
    num_key_value_heads: int = 0,
    sparse_mode: int = 0,
) -> Tensor:
    """Run CATLASS Ascend950 MXFP8 flash attention inference on NPU tensors.

    Source: example 72_ascend950_fp8_mx_flash_attention_infer.

    Args:
        query: FP8 E4M3 query tensor in TND layout
            ``(total_q_tokens, num_heads, head_dim)``.
        key: FP8 E4M3 key cache tensor. For paged KV cache, shape is
            ``(num_blocks, block_size, kv_heads, head_dim)``.
        value: FP8 E4M3 value cache tensor with the same layout as ``key``.
        actual_seq_lengths: Per-batch Q sequence lengths, shape ``(batch,)``.
        actual_seq_lengths_kv: Per-batch KV sequence lengths, shape ``(batch,)``.
        atten_mask: Attention mask tensor. Required when ``sparse_mode == 1``.
        block_table: Paged KV block table, shape ``(batch, max_num_blocks)``.
        q_scale: FP8 E8M0FNU Q scale tensor, shape
            ``(total_q_tokens, ceil_div(num_heads * head_dim, 32))``.
        k_scale: FP8 E8M0FNU K scale tensor, shape
            ``(ceil_div(kv_heads * head_dim, 32), num_blocks * block_size)``.
        v_scale: FP8 E8M0FNU V scale tensor, shape
            ``(ceil_div(num_blocks * block_size, 32), kv_heads * head_dim)``.
        p_scale: Optional float P dequantization scale tensor, shape ``(1,)``.
        input_layout: Only ``TND`` is supported.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        sparse_mode: ``0`` for no mask, ``1`` for chunked causal mask.

    Returns:
        Output tensor with shape ``(total_q_tokens, num_heads, head_dim)``
        in fp16 or bf16.
    """
    return torch.ops.catlass.ascend950_fp8_mx_flash_attention_infer(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        atten_mask,
        block_table,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        input_layout,
        num_heads,
        num_key_value_heads,
        sparse_mode,
    )
