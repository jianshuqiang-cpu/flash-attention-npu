import torch
from torch import Tensor


def ascend950_mx_grouped_matmul_slice_m(
    mat1: Tensor,
    mat2: Tensor,
    groupList: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    transA: bool = False,
    transB: bool = False,
    enable_aswt: bool = False,
    enable_preload: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 MX grouped matmul Slice-M on NPU tensors.

    Source: example 55_ascend950_mx_grouped_matmul_slice_m.

    Computes grouped matmul with MX (Microscaling) quantized inputs.
    A and B are MX-quantized (float8_e4m3/float8_e5m2/float4_e2m1fn_x2)
    with block scales (float8_e8m0fnu). Output is FP32.

    Args:
        mat1: MX-quantized left input, shape ``(M, K)``.
        mat2: MX-quantized right input, shape ``(G, K, N)``.
        groupList: int64 group sizes (non-cumsum), shape ``(G,)``.
        mx_scale_a: MX scale for A (float8_e8m0fnu).
        mx_scale_b: MX scale for B (float8_e8m0fnu), per group.
        transA: Whether to read ``mat1`` as transposed.
        transB: Whether to read ``mat2`` as transposed.
        enable_aswt: Enable the ASWT (tail-split) block scheduler variant.
        enable_preload: Enable the L1 preload mmad pipeline.

    Returns:
        FP32 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_mx_grouped_matmul_slice_m(
        mat1, mat2, groupList, mx_scale_a, mx_scale_b,
        transA, transB, enable_aswt, enable_preload
    )
