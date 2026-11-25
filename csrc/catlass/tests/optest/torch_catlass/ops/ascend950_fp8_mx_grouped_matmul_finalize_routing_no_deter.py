import torch
from torch import Tensor


def ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter(
    mat1: Tensor,
    mat2: Tensor,
    mx_scale_a: Tensor,
    mx_scale_b: Tensor,
    group_list: Tensor,
    logit: Tensor,
    row_index: Tensor,
    bias: Tensor,
    shared_input: Tensor,
    batch: int,
    data_parallel_size: int = 1,
    shared_input_weight: float = 0.0,
    shared_input_offset: int = 0,
    group_list_type: int = 0,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tensor:
    """Run CATLASS Ascend950 MX FP8 grouped matmul + finalize routing (non-deterministic) on NPU tensors.

    Source: example 71_ascend950_fp8_mx_grouped_matmul_finalize_routing (no_deter variant).

    Computes grouped MX-FP8 matmul with logit-weighted scatter-add aggregation,
    optional bias, and optional shared-input contribution (MoE routing post-process).
    Uses non-deterministic scheduling (GemmGroupedAswtTailSplitSwizzle) for
    potentially higher throughput.

    Args:
        mat1: Left input (float8_e4m3fn or float8_e5m2), shape ``(M, K)``.
        mat2: Right input (float8_e4m3fn or float8_e5m2), shape ``(problem_count, N, K)`` when trans_b=True.
        mx_scale_a: MX scale for A (float8_e8m0fnu).
        mx_scale_b: MX scale for B (float8_e8m0fnu).
        group_list: 1-D int64 group boundaries on NPU.
        logit: 1-D float32 routing logits, shape ``(M,)``.
        row_index: 1-D int64 scatter indices, shape ``(M,)``.
        bias: Optional bfloat16 bias ``(problem_count, N)``, or empty tensor.
        shared_input: Optional bfloat16 shared input ``(bsdp, N)``, or empty tensor.
        batch: Number of output rows (scatter target).
        data_parallel_size: Data parallel size (bsdp = batch // dp).
        shared_input_weight: Weight for shared input contribution.
        shared_input_offset: Row offset for shared input placement.
        group_list_type: 0 = prefix-sum, 1 = direct sizes.
        trans_a: Whether mat1 is transposed.
        trans_b: Whether mat2 is transposed.

    Returns:
        FP32 output tensor with shape ``(batch, N)``.
    """
    return torch.ops.catlass.ascend950_fp8_mx_grouped_matmul_finalize_routing_no_deter(
        mat1,
        mat2,
        mx_scale_a,
        mx_scale_b,
        group_list,
        logit,
        row_index,
        bias,
        shared_input,
        trans_a,
        trans_b,
        batch,
        data_parallel_size,
        shared_input_weight,
        shared_input_offset,
        group_list_type,
    )
