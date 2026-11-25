import torch
from torch import Tensor

def group_gemm(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    outDType: str | torch.dtype = torch.float16,
    alpha: float = 1.0,
    beta: float = 0.0,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS grouped GEMM on NPU tensors.

    Source: example 16_group_gemm.

    Computes ``D = alpha * A * B + beta * C`` for each group, where
    groups share the same M and N but have varying K dimensions.

    Args:
        mat1: Concatenated left input matrix ``(K_total, M)`` unless transA.
        mat2: Concatenated right input matrix ``(K_total, N)`` unless transB.
        groupList: 1-D int64 tensor of cumulative K boundaries on NPU device.
        outDType: Output dtype.
        alpha: Scaling factor for matrix product.
        beta: Scaling factor for the output term.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is in CATLASS NZ block format.
        formatB: Whether ``mat2`` is in CATLASS NZ block format.

    Returns:
        Output tensor ``(G, M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.group_gemm(
        mat1, mat2, groupList, outDType, alpha, beta, transA, transB, formatA, formatB
    )
