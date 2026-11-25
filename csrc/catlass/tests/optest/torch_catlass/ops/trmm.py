import torch
from torch import Tensor

_DTYPE_MAP = {
    "float": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
}


def _normalize_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_MAP[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"{dtype} is not supported by torch_catlass.trmm") from exc


def trmm(
    mat1: Tensor,
    mat2: Tensor,
    out_dtype: str | torch.dtype = torch.float32,
    side: int = 0,
    uplo: int = 0,
    trans: int = 0,
    diag: int = 0,
    alpha: float = 1.0,
) -> Tensor:
    """Run CATLASS TRMM on NPU tensors.

    Source: example 76_trmm.

    Args:
        mat1: Left input. When ``side=0`` this is the triangular matrix.
        mat2: Right input. When ``side=1`` this is the triangular matrix.
        out_dtype: Output dtype. The current TRMM kernel supports ``float32``.
        side: ``0`` computes ``op(mat1) @ mat2``; ``1`` computes ``mat1 @ op(mat2)``.
        uplo: ``0`` keeps the lower triangle; ``1`` keeps the upper triangle.
        trans: Whether to transpose the triangular input.
        diag: Diagonal mode. The current TRMM kernel supports only ``0``.
        alpha: Output scaling factor.

    Returns:
        Output tensor with shape ``(M, N)`` on the active NPU device.
    """
    return torch.ops.catlass.trmm(
        mat1, mat2, _normalize_dtype(out_dtype), side, uplo, trans, diag, alpha
    )
