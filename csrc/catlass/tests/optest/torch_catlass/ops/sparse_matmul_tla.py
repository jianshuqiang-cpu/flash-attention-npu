import torch
from torch import Tensor


def sparse_matmul_tla(
    mat1: Tensor,
    mat2: Tensor,
    idx: Tensor,
    outDType: str | torch.dtype = torch.int32,
    transA: bool = False,
    transB: bool = True,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS 4:2 sparse matmul (TLA) on NPU tensors.

    Source: example 41_sparse_matmul_tla.

    Computes ``C = A @ sparse(B)`` where B follows the 4:2 sparse pattern
    (only half the elements stored) and ``idx`` is the index matrix that
    encodes which elements are retained.

    Args:
        mat1: Left input matrix (int8). Shape ``(M, K)``.
        mat2: Right input matrix (int8, sparse-compressed). Shape
            ``(K/2, N)`` for transB=True, as only half the K elements
            are stored in the 4:2 sparse format.
        idx: Sparse index matrix (uint8). Shape ``((N*K+7)//8,)``.
        outDType: Output dtype. Default ``int32``.
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
    return torch.ops.catlass.sparse_matmul_tla(
        mat1, mat2, idx, outDType, transA, transB, formatA, formatB
    )
