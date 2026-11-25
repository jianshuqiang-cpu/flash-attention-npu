import torch
from torch import Tensor


def ascend950_basic_matmul_gemv(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 basic matmul GEMV on NPU tensors.

    Source: example 50_ascend950_basic_matmul_gemv.

    Computes C = A @ B where A uses VectorLayout (GEMV pattern, M is small).
    All inputs and outputs are float32.

    Args:
        mat1: Left input matrix (float32). Shape ``(M, K)`` unless ``transA``.
        mat2: Right input matrix (float32). Shape ``(K, N)`` unless ``transB``.
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
    return torch.ops.catlass.ascend950_basic_matmul_gemv(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
