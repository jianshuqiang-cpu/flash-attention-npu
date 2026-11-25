import torch
from torch import Tensor


def ascend950_multi_core_splitk_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Ascend950 multi-core split-K matmul (example 68).

    Uses the MultiCoreSplitkMatmulTla kernel that splits the K dimension
    to create more work-blocks, enabling higher AI core utilization when
    ``CeilDiv(M, m1) * CeilDiv(N, n1) <= CORES / 2``.

    Args:
        mat1: left operand ``(M, K)`` or ``(K, M)`` when transA.
        mat2: right operand ``(K, N)`` or ``(N, K)`` when transB.
        outDType: output data type (``"float32"`` or ``torch.float32``).
        transA:  ``True`` if mat1 is column-major (shape K × M).
        transB:  ``True`` if mat2 is column-major (shape N × K).
        formatA: ``True`` if A uses Ascend ND-format.
        formatB: ``True`` if B uses Ascend ND-format.

    Returns:
        Tensor of shape ``(M, N)`` with dtype ``outDType`` on NPU.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.ascend950_multi_core_splitk_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )