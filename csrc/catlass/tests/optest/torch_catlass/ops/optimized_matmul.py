import torch
from torch import Tensor

def optimized_matmul(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float16,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS optimized matmul on NPU tensors.

    Source: example 06_optimized_matmul.

    Args:
        mat1: Left input matrix. Shape ``(M, K)`` unless ``transA`` is true.
        mat2: Right input matrix. Shape ``(K, N)`` unless ``transB`` is true.
        outDType: Output dtype. Accepted strings are ``float16``, ``float32``
            and ``bf16``/``bfloat16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is stored in the CATLASS NZ block format.
        formatB: Whether ``mat2`` is stored in the CATLASS NZ block format.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.optimized_matmul(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
