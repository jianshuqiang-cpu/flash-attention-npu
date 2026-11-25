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


def quant_optimized_matmul_tla(
    mat1: Tensor,
    mat2: Tensor,
    scale: Tensor,
    per_token_scale: Tensor,
    outDType: str | torch.dtype = torch.bfloat16,
    transA: bool = False,
    transB: bool = True,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS quantized optimized matmul (TLA) on NPU tensors.

    Source: example 42_quant_optimized_matmul_tla.

    Computes ``D = per_token_scale * (mat1 @ mat2) * scale`` where mat1 and
    mat2 are int8 quantized matrices, scale is a per-column dequantization
    factor and per_token_scale is a per-row dequantization factor.

    Args:
        mat1: Left input matrix (int8). Shape is ``(M, K)`` unless
            ``transA`` is true, in which case shape is ``(K, M)``.
        mat2: Right input matrix (int8). Shape is ``(K, N)`` unless
            ``transB`` is true, in which case shape is ``(N, K)``.
        scale: Per-column scale tensor (float32). Shape ``(1, N)``.
        per_token_scale: Per-row scale tensor (float32). Shape ``(1, M)``.
        outDType: Output dtype. Accepted strings are ``bfloat16``,
            ``float16`` and ``float32``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is stored in the CATLASS NZ block format.
        formatB: Whether ``mat2`` is stored in the CATLASS NZ block format.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.quant_optimized_matmul_tla(
        mat1, mat2, scale, per_token_scale, outDType, transA, transB, formatA, formatB
    )
