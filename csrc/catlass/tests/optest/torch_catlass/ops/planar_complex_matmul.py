import torch
from torch import Tensor


def planar_complex_matmul(
    a_real: Tensor,
    a_imag: Tensor,
    b_real: Tensor,
    b_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run CATLASS planar complex matrix multiplication on NPU tensors.

    Source: example 77_planar_complex_matmul.

    Args:
        a_real: Real part of the left matrix, shape ``(M, K)``, dtype fp16.
        a_imag: Imaginary part of the left matrix, shape ``(M, K)``, dtype fp16.
        b_real: Real part of the right matrix, shape ``(K, N)``, dtype fp16.
        b_imag: Imaginary part of the right matrix, shape ``(K, N)``, dtype fp16.

    Returns:
        ``(c_real, c_imag)`` tensors with shape ``(M, N)`` and dtype fp32.
    """
    return torch.ops.catlass.planar_complex_matmul(
        a_real, a_imag, b_real.t().contiguous(), b_imag.t().contiguous()
    )
