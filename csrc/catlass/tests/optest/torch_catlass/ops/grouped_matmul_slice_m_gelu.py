import torch
from torch import Tensor


def grouped_matmul_slice_m_gelu(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
) -> Tensor:
    """Run CATLASS grouped matmul Slice-M Gelu on NPU tensors.

    Compute: ''D = GELU(A @ B)'' for grouped matrices.

    Source: example 80_grouped_matmul_slice_m_gelu.

    fp16 A @ fp16 B grouped matmul with M-slice using TLA.

    Args:
        mat1: Left input matrix. Shape ``(M, K)``, fp16.
        mat2: Right input matrix. Shape ``(G, K, N)``, fp16.
        groupList: 1-D int64 prefix-sum group boundaries on NPU.

    Returns:
        Output tensor ``(M, N)`` on the active NPU device.
    """

    outDType = torch.float16
    transA = False
    transB = False
    useNzA = False
    useNzB = False

    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.grouped_matmul_slice_m_gelu(
        mat1, mat2, groupList, outDType, transA, transB, useNzA, useNzB
    )
