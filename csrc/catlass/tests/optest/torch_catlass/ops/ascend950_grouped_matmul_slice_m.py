import torch
from torch import Tensor


def ascend950_grouped_matmul_slice_m(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    useNzA: bool = False,
    useNzB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 grouped matmul Slice-M (TLA) on NPU tensors.

    Source: example 60_ascend950_grouped_matmul_slice_m.

    fp16 A @ fp16 B grouped matmul with M-slice using Ascend950 TLA.
    Uses GemmIdentityBlockSwizzle with direction based on m/problemCount vs n.

    Args:
        mat1: Left input matrix. Shape ``(M, K)``, fp16.
        mat2: Right input matrix. Shape ``(G, K, N)``, fp16.
        groupList: 1-D int64 prefix-sum group boundaries on NPU.
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
    return torch.ops.catlass.ascend950_grouped_matmul_slice_m(
        mat1, mat2, groupList, outDType, transA, transB, useNzA, useNzB
    )
