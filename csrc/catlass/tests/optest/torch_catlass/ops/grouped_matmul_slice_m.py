import torch
from torch import Tensor

def grouped_matmul_slice_m(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS grouped matmul (M-slice) on NPU tensors.

    Source: example 02_grouped_matmul_slice_m.

    A is a single 2-D matrix ``(M, K)`` shared across all groups.
    B is a batched 3-D tensor ``(G, K, N)`` (or ``(G, N, K)`` if transB).
    groupList is a 1-D int64 prefix-sum array of length G on CPU.

    The output has shape ``(M, N)`` where M equals the sum of groupList.

    Args:
        mat1: Left input matrix. Shape ``(M, K)``.
        mat2: Right input matrix. Shape ``(G, K, N)``.
        groupList: 1-D int64 prefix-sum group boundaries on CPU.
        outDType: Output dtype.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        useNzA: Whether ``mat1`` uses CATLASS NZ block layout.
        useNzB: Whether ``mat2`` uses CATLASS NZ block layout.

    Returns:
        Output tensor ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_m(
        mat1, mat2, groupList, outDType, transA, transB, useNzA, useNzB
    )
