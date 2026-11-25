from typing import Optional

import torch
from torch import Tensor


def ascend950_matmul_full_dequant(
    mat1: Tensor,
    mat2: Tensor,
    x1Scale: Optional[Tensor] = None,
    x2Scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
    x1QuantMode: str = "per_token",
    x2QuantMode: str = "per_channel",
) -> Tensor:
    """Run CATLASS Ascend950 matmul full dequant on NPU tensors.

    Source: example 57_ascend950_matmul_full_dequant.

    Computes dequantized output from int8 inputs with configurable quantization
    modes for both A (x1) and B (x2), plus optional bias.

    Supported quant mode combinations:
      - x1: per_token, per_tensor, default
      - x2: per_channel, per_tensor, default

    Args:
        mat1: Left input matrix (int8). Shape ``(M, K)`` unless ``transA``.
        mat2: Right input matrix (int8). Shape ``(K, N)`` unless ``transB``.
        x1Scale: Scale for A. Shape depends on x1QuantMode:
            per_tensor: scalar, per_token: ``(M,)``, default: None.
        x2Scale: Scale for B. Shape depends on x2QuantMode:
            per_tensor: scalar, per_channel: ``(N,)``, default: None.
        bias: Optional bias vector (float32). Shape ``(N,)``.
        outDType: Output dtype. Accepted strings are ``float16``, ``bfloat16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is stored in the CATLASS NZ block format.
        formatB: Whether ``mat2`` is stored in the CATLASS NZ block format.
        x1QuantMode: Quantization mode for A. One of ``per_token``,
            ``per_tensor``, ``default``.
        x2QuantMode: Quantization mode for B. One of ``per_channel``,
            ``per_tensor``, ``default``.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_matmul_full_dequant(
        mat1, mat2, x1Scale, x2Scale, bias,
        outDType, transA, transB, formatA, formatB,
        x1QuantMode, x2QuantMode
    )
