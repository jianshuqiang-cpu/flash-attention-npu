import torch
from torch import Tensor

def batched_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS batched matrix multiplication on NPU tensors.

    Source: example 01_batched_matmul.

    Both inputs are 3-D: ``mat1 (B, M, K)``, ``mat2 (B, K, N)``.
    Output is ``(B, M, N)``.

    Args:
        mat1: Left input batch. Shape ``(B, M, K)``.
        mat2: Right input batch. Shape ``(B, K, N)``.
        outDType: Output dtype.
        transA: Whether to transpose each matrix in ``mat1``.
        transB: Whether to transpose each matrix in ``mat2``.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output tensor ``(B, M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.batched_matmul(
        mat1, mat2, outDType, transA, transB, useNzA, useNzB
    )
