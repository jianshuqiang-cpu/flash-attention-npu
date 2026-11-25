import torch
from torch import Tensor


def ascend950_grouped_matmul_slice_m_per_token_dequant(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    scale: Tensor,
    perTokenScale: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 grouped matmul Slice-M with per-token dequant (TLA).

    Source: example 47_ascend950_grouped_matmul_slice_m_per_token_dequant.

    int8 A @ int8 B, dequantized by scale and perTokenScale, output fp16.
    Uses Ascend950 TLA with BlockSchedulerAswt and epilogue per-token dequant.

    Args:
        mat1: int8 ``(M, K)``.
        mat2: int8 ``(G, K, N)``.
        groupList: int64 group boundaries on NPU.
        scale: per-column scale (float, shape ``(G, N)`` or ``(G*N,)``).
        perTokenScale: per-token scale (float, shape ``(M,)``).
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output ``(M, N)`` fp16.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_grouped_matmul_slice_m_per_token_dequant(
        mat1, mat2, groupList, scale, perTokenScale, outDType,
        transA, transB, useNzA, useNzB
    )
