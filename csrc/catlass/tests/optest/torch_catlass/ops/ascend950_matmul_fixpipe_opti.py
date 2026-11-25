import torch
from torch import Tensor


def ascend950_matmul_fixpipe_opti(
    mat1: Tensor,
    mat2: Tensor,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 matmul fixpipe optimization on NPU tensors.

    Source: example 46_ascend950_matmul_fixpipe_opti.

    Uses Fixpipe epilogue to move L0C results to UB efficiently with
    dual destination control support. Computes C = A @ B with half inputs
    and float output.

    Args:
        mat1: Left input matrix (float16). Shape ``(M, K)`` unless ``transA``.
        mat2: Right input matrix (float16). Shape ``(K, N)`` unless ``transB``.
        outDType: Output dtype. Accepted strings are ``float16``, ``float32``
            and ``bf16``/``bfloat16``.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed (ColumnMajor layout). For
            standard PyTorch row-major ``(K, N)`` inputs, leave this ``False``.
            When ``True``, ``mat2`` should be stored as ``(N, K)`` row-major.
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
    return torch.ops.catlass.ascend950_matmul_fixpipe_opti(
        mat1, mat2, outDType, transA, transB, formatA, formatB
    )
