import torch
from torch import Tensor

def grouped_matmul_slice_k(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS grouped matmul (K-slice) on NPU tensors.

    Source: example 05_grouped_matmul_slice_k.

    A and B are single 2-D matrices. A is ``(M, K)`` (or ``(K, M)`` if transA).
    B is ``(K, N)`` (or ``(N, K)`` if transB).
    groupList is a 1-D int64 prefix-sum array of length G on CPU.

    The output has shape ``(G, M, N)`` where G is the number of groups.

    Args:
        mat1: Left input matrix. Shape ``(M, K)``.
        mat2: Right input matrix. Shape ``(K, N)``.
        groupList: 1-D int64 prefix-sum group boundaries on CPU.
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output tensor ``(G, M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_k(
        mat1, mat2, groupList, outDType, transA, transB, useNzA, useNzB
    )
