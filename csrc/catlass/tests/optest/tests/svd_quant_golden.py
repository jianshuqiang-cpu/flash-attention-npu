# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

"""Golden helpers aligned with examples/61_ascend950_svd_quant_matmul/gen_data.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from mx_golden import _pack_fp4_nibbles

_BLOCK_SIZE = 32

# ─────────────────────────────────────────────────────────────
#  NPU-strict _quantize_to_fp4_lut  (ties-to-even + signbit)
# ─────────────────────────────────────────────────────────────

_FP4_LUT = {
    "E2M1": torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    ),
}


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    clamped = values.clamp(-6.0, 6.0)
    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    min_dist = distances.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

    # NPU CAST_RINT 行为: 符号零 + ties-to-even mantissa。
    # 1) ±0.0: 两者都是 even-mantissa，按原始值的 signbit 选择符号零
    signbit = torch.signbit(values)
    tie_0_8 = (distances[..., 0] == distances[..., 8]) & (distances[..., 0] == min_dist)
    indices = torch.where(tie_0_8 & signbit, torch.full_like(indices, 8), indices)

    # 2) 其他中点: odd-mantissa (idx%2==1) 若与相邻 even-mantissa 等距，切换到后者
    is_odd = (indices & 1) == 1
    next_idx = indices + 1
    next_dist = distances.gather(-1, next_idx.clamp(max=15).unsqueeze(-1)).squeeze(-1)
    tie_next = is_odd & (indices < 15) & (next_dist == min_dist)
    indices = torch.where(tie_next, indices + 1, indices)

    quantized = lut[indices]
    return quantized, indices.to(torch.uint8)
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127


def _case_random_seed(*values) -> int:
    seed = 0
    for value in values:
        if isinstance(value, float):
            seed = (seed * 1_000_003 + int(round(value * 1_000))) & 0x7FFFFFFF
        else:
            seed = (seed * 1_000_003 + int(value)) & 0x7FFFFFFF
    return seed


def _set_case_random_seed(*values) -> int:
    seed = _case_random_seed(*values)
    torch.manual_seed(seed)
    return seed


@dataclass
class ErrorMetrics:
    passed: bool
    mare_ratio: float
    mere_ratio: float
    rmse_ratio: float


def _quantize_fp4_with_qmax(
    matrix: torch.Tensor, axis: int, is_normal_matrix: bool = True, qmax: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        qt, st, dt = _quantize_fp4_with_qmax(matrix.t().contiguous(), 1, is_normal_matrix, qmax)
        return qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous()

    m, n = matrix.shape
    padded_n = ((n + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    num_blocks = padded_n // _BLOCK_SIZE
    padded = matrix if padded_n == n else torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
    if padded_n != n:
        padded[:, :n] = matrix
    blocks = padded.view(m, num_blocks, _BLOCK_SIZE)
    max_abs = blocks.abs().amax(dim=-1)
    if is_normal_matrix:
        exp = torch.floor(torch.log2(max_abs.clamp(min=_EPSILON))) - 2.0
    else:
        qmaxInv = torch.tensor(1.0 / qmax, dtype=matrix.dtype, device=matrix.device)
        exp = torch.ceil(torch.log2(torch.clamp(max_abs, min=_EPSILON) * qmaxInv))
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp).clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)
    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, "E2M1")
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)
    quantized = quantized_blocks.reshape(m, padded_n)[:, :n].contiguous()
    dequantized = dequant_blocks.reshape(m, padded_n)[:, :n].contiguous()
    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded
    return quantized, scale, dequantized


def _gen_matrix16_fp4_e2m1(
    matrix: torch.Tensor, axis: int, trans: int, is_normal_matrix: bool = True, qmax: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4_with_qmax(
        matrix, axis, is_normal_matrix, qmax
    )
    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()
    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    packed = _pack_fp4_nibbles(fp4_indices)
    return packed, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def _clone_int8_storage(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.contiguous().flatten()
    return torch.empty(flat.numel(), dtype=torch.int8, device=flat.device).copy_(flat.view(torch.int8))


def compute_error_metrics(
    actual: torch.Tensor,
    cpu_ref: torch.Tensor,
    golden_ref: torch.Tensor,
    mare_threshold: float = 5.0,
    mere_threshold: float = 1.5,
    rmse_threshold: float = 1.5,
) -> ErrorMetrics:
    """Match examples/common/golden/compare_data.hpp ComputeErrorMetrics ratios."""
    eps = 1e-7
    c = actual.detach().cpu().float().flatten()
    cpu = cpu_ref.detach().cpu().float().flatten()
    gold = golden_ref.detach().cpu().float().flatten()
    diff_c = (c - gold).abs()
    diff_cpu = (cpu - gold).abs()
    rel_c = diff_c / (gold.abs() + eps)
    rel_cpu = diff_cpu / (gold.abs() + eps)
    mare_c = rel_c.max().item()
    mere_c = rel_c.mean().item()
    rmse_c = torch.sqrt((diff_c * diff_c).mean()).item()
    mare_cpu = rel_cpu.max().item()
    mere_cpu = rel_cpu.mean().item()
    rmse_cpu = torch.sqrt((diff_cpu * diff_cpu).mean()).item()
    mare_ratio = mare_c / mare_cpu if mare_cpu > 0 else 0.0
    mere_ratio = mere_c / mere_cpu if mere_cpu > 0 else 0.0
    rmse_ratio = rmse_c / rmse_cpu if rmse_cpu > 0 else 0.0
    passed = mare_ratio <= mare_threshold and mere_ratio <= mere_threshold and rmse_ratio <= rmse_threshold
    return ErrorMetrics(passed, mare_ratio, mere_ratio, rmse_ratio)


def prepare_svd_quant_matmul_inputs(
    m: int,
    n: int,
    k: int,
    r: int,
    qmax: float = 8.0,
    device: str = "npu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs and references (y_cpu, y_golden) aligned with example 61 gen_data.py."""
    _set_case_random_seed(61, m, n, k, r, int(round(qmax * 1000)))
    dtype16 = torch.float16

    x_fp16 = torch.randn(m, k, dtype=dtype16)
    svd1 = torch.randn(k, r, dtype=dtype16)
    svd2 = torch.randn(r, n, dtype=dtype16)

    smooth_scale = 1.0 + torch.randn(k, dtype=torch.float32) * 0.1
    smooth_scale_inv = (1.0 / smooth_scale).to(dtype16)
    smooth_x = x_fp16 @ torch.diag(smooth_scale_inv)
    smooth_x_fp32 = x_fp16.to(torch.float32) @ torch.diag(1.0 / smooth_scale)

    _, _, deq_smooth_x_fp32 = _gen_matrix16_fp4_e2m1(
        smooth_x, axis=1, trans=0, is_normal_matrix=False, qmax=qmax
    )

    w_fp16 = torch.randn(k, n, dtype=dtype16) * 0.1
    w_packed, w_scale, deq_w_fp32 = _gen_matrix16_fp4_e2m1(w_fp16, axis=0, trans=1, is_normal_matrix=True)
    w_scale = w_scale.reshape(w_scale.shape[0] // 2, 2, w_scale.shape[1]).permute(2, 0, 1).contiguous()

    c1_fp32 = smooth_x.to(torch.float32) @ svd1.to(torch.float32)
    c1_fp16 = c1_fp32.to(dtype16)
    y_cpu = c1_fp16.to(torch.float32) @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32
    y_golden = smooth_x_fp32 @ svd1.to(torch.float32) @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32

    svd1 = svd1.permute(1, 0).contiguous().permute(1, 0)
    svd2 = svd2.permute(1, 0).contiguous().permute(1, 0)
    w_int8 = _clone_int8_storage(w_packed)

    if device == "npu":
        x_fp16 = x_fp16.contiguous().npu()
        svd1 = svd1.npu()
        svd2 = svd2.npu()
        w_int8 = w_int8.contiguous().npu()
        w_scale = w_scale.contiguous().npu()
        smooth_scale_inv = smooth_scale_inv.contiguous().npu()

    return x_fp16, svd1, svd2, w_int8, w_scale, smooth_scale_inv, y_cpu, y_golden
