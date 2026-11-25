import torch
from torch import Tensor

def gemm(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    alpha: float = 1.0,
    beta: float = 0.0,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS GEMM with alpha/beta scaling on NPU tensors.

    Source: example 15_gemm.

    Computes ``D = alpha * A * B + beta * C`` where ``C`` is the output tensor
    pre-filled with zeros (or an initial value provided by the caller).

    Args:
        mat1: Left input matrix ``(M, K)`` unless ``transA`` is true.
        mat2: Right input matrix ``(K, N)`` unless ``transB`` is true.
        outDType: Output dtype. Accepted strings are ``float16``, ``float32``.
        alpha: Scaling factor for matrix product.
        beta: Scaling factor for the output (residual) term.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        formatA: Whether ``mat1`` is in CATLASS NZ block format.
        formatB: Whether ``mat2`` is in CATLASS NZ block format.

    Returns:
        Output tensor ``(M, N)`` on the active NPU device.
    """
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.gemm(
        mat1, mat2, outDType, alpha, beta, transA, transB, formatA, formatB
    )
