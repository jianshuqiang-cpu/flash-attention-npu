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


def broadcast_matmul_perblock_quant(
    a: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run CATLASS broadcast matmul with per-block quantization on NPU tensors.

    Source: example 62_ascend950_broadcast_matmul_perblock_quant.

    Computes matrix multiplication where tensor ``a`` has shape ``(batch, M, K)``
    and tensor ``b`` has shape ``(K, N)``. The result is a ``(batch, M, N)`` output
    with per-block quantization applied.

    Args:
        a: Left input tensor (bfloat16). Shape is ``(batch, M, K)``.
        b: Right input tensor (bfloat16). Shape is ``(K, N)``.

    Returns:
        A tuple of (output tensor, scale tensor):
        - output: Tensor with shape ``(batch, M, N)`` on NPU, dtype float8_e4m3fn.
        - scale: Per-block scale tensor with shape ``(batch,)`` on NPU, dtype float32.
    """
    return torch.ops.catlass.broadcast_matmul_perblock_quant(a, b)
