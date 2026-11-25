import torch
from torch import Tensor


def w4a8_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS W4A8 matmul on NPU tensors.

    Source: example 32_w4a8_matmul.

    Computes ``C = A_int8 @ B_int4_deq * 1.5`` where B is an int4
    weight matrix dequantized to int8 via prologue, then per-tensor scaled.

    Args:
        mat1: Left input matrix (int8). Shape ``(M, K)``.
        mat2: Right input matrix (int4 packed as int8). Shape ``(K, N)``.
        outDType: Output dtype. Default ``float16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is in CATLASS NZ block format.
        formatB: Whether ``mat2`` is in CATLASS NZ block format.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.w4a8_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
