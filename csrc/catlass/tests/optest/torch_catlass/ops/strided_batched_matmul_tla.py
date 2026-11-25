import torch
from torch import Tensor


def strided_batched_matmul_tla(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
    batchTransA: bool = False,
    batchTransB: bool = False,
) -> Tensor:
    """Run CATLASS strided batched matmul (TLA) on NPU tensors.

    Source: example 45_strided_batched_matmul_tla.

    Computes ``C[b] = A[b] @ B[b]`` for each batch ``b`` using strided
    memory access, where A and B are 3D tensors with batch as the first
    dimension.

    Args:
        mat1: Left input matrix (float16). Shape ``(B, M, K)``.
        mat2: Right input matrix (float16). Shape ``(B, K, N)``.
        outDType: Output dtype. Default ``float16``.
        transA: Whether to read each ``mat1[b]`` as transposed.
        transB: Whether to read each ``mat2[b]`` as transposed.
        formatA: Whether ``mat1`` is in CATLASS NZ block format.
        formatB: Whether ``mat2`` is in CATLASS NZ block format.
        batchTransA: Whether the batch axis of ``mat1`` sits on the
            middle dimension (``(M, B, K)``) instead of the first.
        batchTransB: Whether the batch axis of ``mat2`` sits on the
            middle dimension (``(K, B, N)``) instead of the first.

    Returns:
        Output tensor with shape ``(B, M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.strided_batched_matmul_tla(
        mat1, mat2, outDType, transA, transB, formatA, formatB,
        batchTransA, batchTransB,
    )
