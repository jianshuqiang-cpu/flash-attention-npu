import torch
from torch import Tensor


def ascend950_quant_matmul_per_group_per_block_tla(
    mat1: Tensor,
    mat2: Tensor,
    x1Scale: Tensor,
    x2Scale: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 quant matmul per-group per-block (TLA) on NPU tensors.

    Source: example 51_ascend950_quant_matmul_per_group_per_block_tla.

    Computes dequantized output from int8 inputs with per-group scale for A
    (group size 128) and per-block scale for B (block size 128).

    Args:
        mat1: Left input matrix (int8). Shape ``(M, K)`` unless ``transA``.
        mat2: Right input matrix (int8). Shape ``(K, N)`` unless ``transB``.
        x1Scale: Per-group scale for A (float32). Shape ``(M, CeilDiv(K, 128))``.
        x2Scale: Per-block scale for B (float32). Shape ``(CeilDiv(K, 128), CeilDiv(N, 128))``.
        outDType: Output dtype. Accepted strings are ``float16``, ``bfloat16``.
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
    return torch.ops.catlass.ascend950_quant_matmul_per_group_per_block_tla(
        mat1, mat2, x1Scale, x2Scale, outDType, transA, transB, formatA, formatB
    )
