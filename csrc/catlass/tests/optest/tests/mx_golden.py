# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

"""MX matmul golden reference helpers for optest (aligned with example gen_data.py layout)."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

MX_SCALE_GROUP_NUM = 32
_FP4_BLOCK_SIZE = 32
_FP4_EPSILON = 1e-12
_FP4_MIN_SCALE_EXP = -128
_FP4_MAX_SCALE_EXP = 127

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
}


def _round_up2(v: int) -> int:
    return ((v + 1) // 2) * 2


def mx_scale_k(k: int) -> int:
    return _round_up2(math.ceil(k / MX_SCALE_GROUP_NUM))


def _build_fp4_lut(format_name: str) -> torch.Tensor:
    config = _FP4_FORMATS[format_name]
    exp_bits = int(config["exp_bits"])
    mantissa_bits = int(config["mantissa_bits"])
    bias = float(config["bias"])

    values = []
    for i in range(16):
        sign = (i >> 3) & 0x01
        exp = (i >> mantissa_bits) & ((1 << exp_bits) - 1)
        mantissa = i & ((1 << mantissa_bits) - 1)

        if exp == 0:
            if mantissa == 0:
                value = 0.0
            else:
                value = (mantissa / float(1 << mantissa_bits)) * (2.0 ** (1.0 - bias))
        else:
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (2.0 ** (float(exp) - bias))

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {"E2M1": _build_fp4_lut("E2M1")}


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]
    clamped = values.clamp(min_value, max_value)
    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    return lut[indices], indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )
    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    return (low | high).to(torch.uint8)


def _quantize_fp4_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    padded_n = ((n + block_size - 1) // block_size) * block_size
    num_blocks = padded_n // block_size

    if padded_n != n:
        padded = torch.zeros((m, padded_n), dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(torch.clamp(max_abs, min=_FP4_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    exp = torch.where(max_abs < _FP4_EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_FP4_MIN_SCALE_EXP, _FP4_MAX_SCALE_EXP)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=matrix.device), exp)

    scaled = blocks / scale.unsqueeze(-1)
    quantized_blocks, _ = _quantize_to_fp4_lut(scaled, format_name)
    dequant_blocks = quantized_blocks * scale.unsqueeze(-1)

    quantized = quantized_blocks.reshape(m, padded_n)
    dequantized = dequant_blocks.reshape(m, padded_n)
    if padded_n != n:
        quantized = quantized[:, :n].contiguous()
        dequantized = dequantized[:, :n].contiguous()

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_fp4_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_fp4_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return quantized_t.t().contiguous(), scale_t.t().contiguous(), dequantized_t.t().contiguous()


def _quantize_fp4(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _FP4_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp4_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp4_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def _gen_fp4_e2m1(row: int, col: int, axis: int, trans: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate packed FP4 matrix, MX scale, and dequant FP32 (example 54 gen_data_fp4_e2m1)."""
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4(matrix, "E2M1", axis)

    if trans:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    packed_uint8 = _pack_fp4_nibbles(fp4_indices)
    return packed_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def _packed_uint8_to_fp4(packed_uint8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Materialize packed uint8 nibbles as float4_e2m1fn_x2 with logical shape (rows, cols)."""
    packed = packed_uint8.contiguous().flatten()
    out = torch.empty((rows, cols), dtype=torch.float4_e2m1fn_x2)
    out.view(torch.uint8).flatten()[: packed.numel()].copy_(packed)
    return out


def _quantize_axis_last(
    matrix: torch.Tensor, fp8_dtype: torch.dtype, emax: int, fp8_max: float, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = matrix.shape
    num_blocks = (n + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != n:
        padded = torch.zeros(m, padded_n, dtype=matrix.dtype, device=matrix.device)
        padded[:, :n] = matrix
    else:
        padded = matrix

    blocks = padded.view(m, num_blocks, block_size)
    max_abs = blocks.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(max_abs.clamp(min=1e-12))) - emax
    exp = exp.clamp(-128, 127)
    scale = torch.exp2(exp.to(torch.float32))
    scaled = (blocks / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    quant_fp8 = scaled.to(fp8_dtype)
    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    if padded_n != n:
        quant_fp8 = quant_fp8.reshape(m, padded_n)[:, :n].contiguous()
        dequant = dequant.reshape(m, padded_n)[:, :n].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(m, n)
        dequant = dequant.reshape(m, n)

    padded_blocks = _round_up2(num_blocks)
    if padded_blocks != num_blocks:
        scale_padded = torch.ones(m, padded_blocks, dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale.to(torch.float8_e8m0fnu), dequant


def _quantize_axis_first(
    matrix: torch.Tensor, fp8_dtype: torch.dtype, emax: int, fp8_max: float, block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qt, st, dt = _quantize_axis_last(matrix.t().contiguous(), fp8_dtype, emax, fp8_max, block_size)
    return qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous()


def _move_kernel_inputs_to_npu(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move MX kernel inputs to the current NPU device (call after set_device)."""
    return (
        a.contiguous().npu(),
        b.contiguous().npu(),
        a_scale.contiguous().npu(),
        b_scale.contiguous().npu(),
    )


def prepare_fp8_mx_inputs(
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = True,
    device: str = "npu",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    emax: int = 8,
    fp8_max: float = 448.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MX-FP8 inputs for the given transpose flags (aligned with example 53 layout).

    ``trans_a`` / ``trans_b`` match the ``torch.ops.catlass`` MX matmul API:
    - ``trans_a=False``: ``mat1`` stored as ``(M, K)``; ``trans_a=True``: ``(K, M)``.
    - ``trans_b=True``: ``mat2`` stored as ``(N, K)``; ``trans_b=False``: ``(K, N)``.

    ``fp8_dtype`` / ``emax`` / ``fp8_max`` select the MX-FP8 format
    (E4M3: ``emax=8, fp8_max=448.0``; E5M2: ``emax=15, fp8_max=57344.0``).

    Quantization and reference matmul run on CPU; kernel inputs are moved to NPU
    when ``device="npu"``. ``expected`` stays on CPU for numerical comparison.
    """
    a_fp32 = torch.randn(m, k, dtype=torch.float32) * 10.0 - 5.0
    b_fp32 = torch.randn(k, n, dtype=torch.float32) * 10.0 - 5.0

    a_fp8, a_scale, a_deq = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)
    b_fp8, b_scale, b_deq = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)

    a_scale = a_scale.reshape(m, a_scale.shape[1] // 2, 2).contiguous()
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).permute(2, 0, 1).contiguous()

    if trans_a:
        a_fp8 = a_fp8.t().contiguous()
        a_scale = a_scale.permute(1, 0, 2).contiguous()
    else:
        a_fp8 = a_fp8.contiguous()
    
    if trans_b:
        b_fp8 = b_fp8.t().contiguous()
    else:
        b_fp8 = b_fp8.contiguous()
        b_scale = b_scale.permute(1, 0, 2).contiguous()

    expected = a_deq @ b_deq
    if device == "npu":
        a_fp8, b_fp8, a_scale, b_scale = _move_kernel_inputs_to_npu(a_fp8, b_fp8, a_scale, b_scale)
    return a_fp8, b_fp8, a_scale, b_scale, expected


def prepare_fp8_mx_batch_inputs(
    batch: int, m: int, n: int, k: int, device: str = "npu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build batched MX-FP8 inputs for example 58 (trans_a=0, trans_b=1)."""
    fp8_dtype = torch.float8_e4m3fn
    emax, fp8_max = 8, 448.0

    a_list = []
    b_list = []
    a_scale_list = []
    b_scale_list = []
    expected_list = []

    for _ in range(batch):
        a_fp32 = torch.randn(m, k, dtype=torch.float32) * 10.0 - 5.0
        b_fp32 = torch.randn(k, n, dtype=torch.float32) * 10.0 - 5.0

        a_fp8, a_scale, a_deq = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)
        b_fp8, b_scale, b_deq = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)

        a_list.append(a_fp8.contiguous())
        b_list.append(b_fp8.t().contiguous())
        a_scale_list.append(a_scale.reshape(m, a_scale.shape[1] // 2, 2).contiguous())
        b_scale_list.append(
            b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).permute(2, 0, 1).contiguous()
        )
        expected_list.append(a_deq @ b_deq)

    a = torch.stack(a_list, dim=0)
    b = torch.stack(b_list, dim=0)
    a_scale = torch.stack(a_scale_list, dim=0)
    b_scale = torch.stack(b_scale_list, dim=0)
    expected = torch.stack(expected_list, dim=0)

    if device == "npu":
        a = a.contiguous().npu()
        b = b.contiguous().npu()
        a_scale = a_scale.contiguous().npu()
        b_scale = b_scale.contiguous().npu()
    return a, b, a_scale, b_scale, expected


def prepare_fp8_mx_grouped_slice_m_inputs(
    group_sizes: Sequence[int],
    n: int,
    k: int,
    device: str = "npu",
    trans_b: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build grouped MX-FP8 Slice-M inputs for example 55.

    ``group_sizes`` is the non-cumsum groupList consumed by the kernel. A is
    stored as ``(sum(group_sizes), K)``. B is stored as ``(G, K, N)`` when
    ``trans_b=False`` and ``(G, N, K)`` when ``trans_b=True``.
    """
    group_sizes = tuple(group_sizes)
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    if any(size <= 0 for size in group_sizes):
        raise ValueError(f"group_sizes must be positive, got {group_sizes}")

    fp8_dtype = torch.float8_e4m3fn
    emax, fp8_max = 8, 448.0
    m = sum(group_sizes)

    a_fp32 = torch.randn(m, k, dtype=torch.float32) * 10.0 - 5.0
    a_fp8, a_scale, a_deq = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)
    a_scale = a_scale.reshape(m, a_scale.shape[1] // 2, 2).contiguous()

    b_list = []
    b_scale_list = []
    expected_list = []
    offset = 0
    for group_size in group_sizes:
        b_fp32 = torch.randn(k, n, dtype=torch.float32) * 10.0 - 5.0
        b_fp8, b_scale, b_deq = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)
        b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

        if trans_b:
            b_list.append(b_fp8.t().contiguous())
            b_scale_list.append(b_scale.permute(2, 0, 1).contiguous())
        else:
            b_list.append(b_fp8.contiguous())
            b_scale_list.append(b_scale.permute(0, 2, 1).contiguous())

        expected_list.append(a_deq[offset : offset + group_size] @ b_deq)
        offset += group_size

    b = torch.stack(b_list, dim=0)
    b_scale = torch.stack(b_scale_list, dim=0)
    expected = torch.cat(expected_list, dim=0)

    if device == "npu":
        a_fp8, b, a_scale, b_scale = _move_kernel_inputs_to_npu(a_fp8, b, a_scale, b_scale)
    return a_fp8, b, a_scale, b_scale, expected


def prepare_fp4_mx_inputs(
    m: int, n: int, k: int, device: str = "npu", trans_a: bool = False, trans_b: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MX-FP4 inputs for trans_a=0, trans_b=1 (aligned with example 54 gen_data.py).

    Quantization and reference matmul run on CPU; kernel inputs are moved to NPU
    when ``device="npu"``. ``expected`` stays on CPU for numerical comparison.
    """
    a_packed, a_scale, a_deq = _gen_fp4_e2m1(m, k, axis=1, trans=trans_a)
    b_packed, b_scale, b_deq = _gen_fp4_e2m1(k, n, axis=0, trans=trans_b)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2).contiguous()
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1]).contiguous()
    if trans_a:
        a_scale = a_scale.permute(1, 0, 2).contiguous()
    if trans_b:
        b_scale = b_scale.permute(2, 0, 1).contiguous()
    else:
        b_scale = b_scale.permute(0, 2, 1).contiguous()

    if trans_a:
        a_fp4 = _packed_uint8_to_fp4(a_packed, k, m)
    else:
        a_fp4 = _packed_uint8_to_fp4(a_packed, m, k)

    if trans_b:
        b_fp4 = _packed_uint8_to_fp4(b_packed, n, k)
    else:
        b_fp4 = _packed_uint8_to_fp4(b_packed, k, n)

    expected = a_deq @ b_deq
    if device == "npu":
        a_fp4, b_fp4, a_scale, b_scale = _move_kernel_inputs_to_npu(a_fp4, b_fp4, a_scale, b_scale)
    return a_fp4, b_fp4, a_scale, b_scale, expected


_LEVEL0_BLOCK_SIZE = 512
_LEVEL1_BLOCK_SIZE = 32
_FP4_E2M1_MAX = 6.0
_FP4_E2M1_EMAX = 2
_FP4_E2M1_UNSIGNED_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_FP4_E2M1_LUT = np.concatenate(
    [_FP4_E2M1_UNSIGNED_LUT, -_FP4_E2M1_UNSIGNED_LUT]
).astype(np.float32)


def _fp4_rint_codes(values: np.ndarray) -> np.ndarray:
    values_f64 = np.asarray(values, dtype=np.float32).reshape(-1).astype(np.float64)
    sign = np.signbit(values_f64).astype(np.uint8)
    abs_values = np.minimum(np.abs(values_f64), _FP4_E2M1_MAX)
    hi = np.searchsorted(_FP4_E2M1_UNSIGNED_LUT, abs_values, side="right")
    hi = np.clip(hi, 1, len(_FP4_E2M1_UNSIGNED_LUT) - 1)
    lo = hi - 1
    lo_val = _FP4_E2M1_UNSIGNED_LUT[lo]
    hi_val = _FP4_E2M1_UNSIGNED_LUT[hi]
    mid = (lo_val + hi_val) / 2.0
    pick_lo = (abs_values < mid) | ((abs_values == mid) & ((lo & 1) == 0))
    chosen = np.where(pick_lo, lo, hi).astype(np.uint8)
    return (chosen | (sign << 3)).reshape(np.asarray(values).shape).astype(np.uint8)


def _pack_fp4_codes(codes: np.ndarray) -> np.ndarray:
    if codes.shape[-1] % 2 != 0:
        pad = np.zeros((*codes.shape[:-1], 1), dtype=np.uint8)
        codes = np.concatenate([codes, pad], axis=-1)
    pair = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).astype(np.uint8)


def _unpack_fp4_codes(packed: np.ndarray, k: int) -> np.ndarray:
    low = packed & np.uint8(0x0F)
    high = (packed >> np.uint8(4)) & np.uint8(0x0F)
    unpacked = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :k]


def _compute_e8m0_scale(max_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zero_mask = max_abs < 1e-12
    safe_abs = np.where(zero_mask, 1.0, max_abs)
    exp = np.floor(np.log2(safe_abs)).astype(np.int64) - _FP4_E2M1_EMAX
    exp = np.clip(exp, -128, 127)
    scale_float = np.where(zero_mask, np.float64(1.0), np.ldexp(np.float64(1.0), exp.astype(np.int32)))
    scale_u8 = np.where(zero_mask, np.uint8(127), (exp + 127).astype(np.uint8))
    return scale_float, scale_u8.astype(np.uint8)


def _dual_level_quantize_rows(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    rows, k = matrix.shape
    if k % 2 != 0:
        raise ValueError(f"K must be even for FP4 packing: k={k}")

    num_l0 = math.ceil(k / _LEVEL0_BLOCK_SIZE)
    num_l1 = math.ceil(k / _LEVEL1_BLOCK_SIZE)
    k_padded = num_l0 * _LEVEL0_BLOCK_SIZE
    num_l1_padded = k_padded // _LEVEL1_BLOCK_SIZE

    work = np.zeros((rows, k_padded), dtype=np.float32)
    work[:, :k] = matrix

    l0 = work.reshape(rows, num_l0, _LEVEL0_BLOCK_SIZE)
    max_abs0 = np.max(np.abs(l0), axis=-1)
    scale0 = np.where(max_abs0 < 1e-12, np.float32(1.0), (max_abs0 / _FP4_E2M1_MAX).astype(np.float32))
    temp = (work / np.repeat(scale0, _LEVEL0_BLOCK_SIZE, axis=1)).astype(np.float16).astype(np.float32)

    l1 = temp.reshape(rows, num_l1_padded, _LEVEL1_BLOCK_SIZE)
    max_abs1 = np.max(np.abs(l1), axis=-1)
    scale1_float, scale1_u8 = _compute_e8m0_scale(max_abs1)
    scaled = (l1.astype(np.float64) / scale1_float[..., np.newaxis]).reshape(rows, k_padded)[:, :k]
    codes = _fp4_rint_codes(scaled)
    return _pack_fp4_codes(codes), scale1_u8[:, :num_l1]


def _dequantize_mx_only(quant_data: np.ndarray, scale1: np.ndarray, k: int) -> np.ndarray:
    codes = _unpack_fp4_codes(quant_data, k)
    fp4_vals = _FP4_E2M1_LUT[codes]
    exp = scale1.astype(np.int32) - 127
    scale = np.ldexp(np.float64(1.0), exp)
    scale_expanded = np.repeat(scale, _LEVEL1_BLOCK_SIZE, axis=1)[:, :k]
    return (fp4_vals.astype(np.float64) * scale_expanded).astype(np.float32)


def _bf16_round_float32(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).to(torch.float32)


def prepare_dual_level_quant_mx_batch_inputs(
    batch: int, m: int, n: int, k: int, device: str = "npu", dtype: torch.dtype = torch.float16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fp16/bf16 inputs and golden for example 63."""
    if k % 2 != 0:
        raise ValueError(f"K must be even for FP4 packing: k={k}")

    a_list = []
    b_list = []
    expected_list = []

    for _ in range(batch):
        a = (torch.randn(m, k, dtype=torch.float32) * 2.0).to(dtype)
        b = (torch.randn(k, n, dtype=torch.float32) * 2.0).to(dtype)

        a_np = a.to(torch.float16 if dtype == torch.float16 else torch.bfloat16).to(torch.float32).numpy()
        b_np = b.to(torch.float16 if dtype == torch.float16 else torch.bfloat16).to(torch.float32).numpy()

        quant_a, scale1_a = _dual_level_quantize_rows(a_np)
        quant_b, scale1_b = _dual_level_quantize_rows(b_np.T.copy())
        dequant_a = _dequantize_mx_only(quant_a, scale1_a, k)
        dequant_b = _dequantize_mx_only(quant_b, scale1_b, k).T.copy()
        expected_list.append(_bf16_round_float32(dequant_a @ dequant_b))

        a_list.append(a.contiguous())
        b_list.append(b.t().contiguous())

    a_batch = torch.stack(a_list, dim=0)
    b_batch = torch.stack(b_list, dim=0)
    expected = torch.stack(expected_list, dim=0)

    if device == "npu":
        a_batch = a_batch.contiguous().npu()
        b_batch = b_batch.contiguous().npu()
    return a_batch, b_batch, expected


def prepare_a8w4_mx_inputs(
    m: int, n: int, k: int, device: str = "npu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deprecated wrapper; use a8w4_golden.prepare_a8w4_mx_inputs."""
    from a8w4_golden import prepare_a8w4_mx_inputs as _prepare

    return _prepare(m, n, k, device=device)

def _mx_fp8_quant_output(swiglu_out: torch.Tensor) -> tuple:
    """Replicate gen_data.py quant() for MX FP8 output quantization (block_size=32, axis=col)."""
    MX_BLOCK_SIZE = 32
    MAX_EXP_FOR_BF16 = 0x7F80
    BF16_EXP_BIAS = 0x7F00
    SHR_NUM_FOR_BF16 = 7
    EMAX_SHIFTED = 0x0400
    NAN_CUSTOMIZATION = 0x7F81
    MAX_EXP_FOR_FP8 = 0x00FF
    SPECIAL_EXP_THRESHOLD = 0x0040

    data_bf16 = swiglu_out.to(torch.bfloat16)
    M, N = data_bf16.shape

    data_uint16 = data_bf16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp_field = data_uint16 & MAX_EXP_FOR_BF16

    n_blocks = (N + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    N_padded = n_blocks * MX_BLOCK_SIZE

    if N_padded > N:
        exp_padded = torch.zeros(M, N_padded, dtype=torch.int32)
        exp_padded[:, :N] = exp_field
        data_padded = torch.zeros(M, N_padded, dtype=torch.bfloat16)
        data_padded[:, :N] = data_bf16
    else:
        exp_padded = exp_field
        data_padded = data_bf16

    exp_blocks = exp_padded.reshape(M, n_blocks, MX_BLOCK_SIZE)
    max_exp = exp_blocks.max(dim=-1).values

    cmp_result = max_exp != MAX_EXP_FOR_BF16
    zero_mask = max_exp != 0
    invalid_data_mask = max_exp <= EMAX_SHIFTED

    t_emax_shifted = torch.tensor(EMAX_SHIFTED, dtype=torch.int32)
    t_max_exp_fp8 = torch.tensor(MAX_EXP_FOR_FP8, dtype=torch.int32)
    t_zero = torch.tensor(0, dtype=torch.int32)
    t_nan_cust = torch.tensor(NAN_CUSTOMIZATION, dtype=torch.int32)
    t_special = torch.tensor(SPECIAL_EXP_THRESHOLD, dtype=torch.int32)

    max_exp_clamped = torch.where(invalid_data_mask, t_emax_shifted, max_exp)
    shared_exp = max_exp_clamped - EMAX_SHIFTED
    scale_value = shared_exp >> SHR_NUM_FOR_BF16

    scale_value = torch.where(cmp_result, scale_value, t_max_exp_fp8)
    scale_value = torch.where(zero_mask, scale_value, t_zero)

    special_data_mask = shared_exp == BF16_EXP_BIAS
    half_scale_val = BF16_EXP_BIAS - shared_exp
    half_scale_val = torch.where(cmp_result, half_scale_val, t_nan_cust)
    half_scale_val = torch.where(zero_mask, half_scale_val, t_zero)
    half_scale_val = torch.where(special_data_mask, t_special, half_scale_val)

    half_scale_expanded = half_scale_val.unsqueeze(-1).expand(-1, -1, MX_BLOCK_SIZE).reshape(M, N_padded)
    half_scale_bf16 = half_scale_expanded.to(torch.int16).contiguous().view(torch.bfloat16)

    scaled_data = data_padded * half_scale_bf16
    scaled_data_f32 = scaled_data.to(torch.float32)[:, :N]

    scaled_data_f32 = torch.clamp(scaled_data_f32, -448.0, 448.0)
    quant_out = scaled_data_f32.to(torch.float8_e4m3fn)

    return quant_out, scale_value.to(torch.uint8)


def prepare_grouped_mx_swiglu_quant_inputs(
    m: int, n: int, k: int, group_count: int, device: str = "npu"
) -> tuple:
    """Build inputs and golden reference for example 66 grouped MX FP8 matmul + SwiGLU + MX quant.

    Aligned with examples/65_*/gen_data.py golden_compute().

    Returns:
        (a_fp8, b_fp8_trans, a_scale, b_scale_trans, group_list, expected_q, expected_q_scale)
        where kernel inputs are on ``device`` and golden outputs are on CPU.
    """
    torch.manual_seed(42)

    fp8_dtype = torch.float8_e4m3fn
    emax, fp8_max = 8, 448.0

    a_fp32 = torch.randn(m, k, dtype=torch.float32) * 10.0
    a_fp8, a_scale_raw, _ = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)

    k0 = mx_scale_k(k)

    b_fp8_list = []
    b_deq_list = []
    b_scale_raw_list = []
    for _ in range(group_count):
        b_fp32 = torch.randn(k, n, dtype=torch.float32) * 10.0
        b_fp8, b_scale_raw, b_deq = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)
        b_fp8_list.append(b_fp8)
        b_deq_list.append(b_deq)
        b_scale_raw_list.append(b_scale_raw)

    a_scale_packed = a_scale_raw.reshape(m, k0 // 2, 2).contiguous()

    b_fp8_trans_list = []
    b_scale_trans_list = []
    for i in range(group_count):
        b_fp8_trans = b_fp8_list[i].t().contiguous()
        b_fp8_trans_list.append(b_fp8_trans)
        b_sr = b_scale_raw_list[i]
        b_scale_reshaped = b_sr.reshape(b_sr.shape[0] // 2, 2, b_sr.shape[1])
        b_scale_t = b_scale_reshaped.permute(2, 0, 1).contiguous()
        b_scale_trans_list.append(b_scale_t)

    b_fp8_trans_all = torch.stack(b_fp8_trans_list, dim=0)
    b_scale_trans_all = torch.stack(b_scale_trans_list, dim=0)

    a_fp8_npu = a_fp8.to(fp8_dtype)
    b_fp8_trans_npu = b_fp8_trans_all.to(fp8_dtype)
    a_scale_npu = a_scale_packed.to(torch.float8_e8m0fnu)
    b_scale_trans_npu = b_scale_trans_all.to(torch.float8_e8m0fnu)

    group_size = m // group_count
    group_list_noncum = torch.tensor([group_size] * group_count, dtype=torch.int64)

    if device == "npu":
        a_fp8_npu = a_fp8_npu.contiguous().npu()
        b_fp8_trans_npu = b_fp8_trans_npu.contiguous().npu()
        a_scale_npu = a_scale_npu.contiguous().npu()
        b_scale_trans_npu = b_scale_trans_npu.contiguous().npu()
        group_list_noncum = group_list_noncum.npu()

    group_list_cum = torch.cumsum(group_list_noncum, dim=0)

    x = a_fp8.to(torch.float32)
    x_s = a_scale_raw.reshape(m, k0).repeat_interleave(32, dim=-1)
    if k0 * 32 > k:
        x_padded = torch.zeros(m, k0 * 32, dtype=torch.float32)
        x_padded[:, :k] = x
        x = x_padded
    x1 = x.float() * x_s.float()

    grouped_q = []
    grouped_q_scale = []
    for i in range(group_count):
        w = b_fp8_list[i].to(torch.float32)
        w_s_raw = b_scale_raw_list[i]
        w_s = w_s_raw.reshape(k0, n).repeat_interleave(32, dim=0)
        if k0 * 32 > k:
            w_padded = torch.zeros(k0 * 32, n, dtype=torch.float32)
            w_padded[:k, :] = w
            w = w_padded
        x2 = w.float() * w_s.float()

        start = 0 if i == 0 else group_list_cum[i - 1].item()
        end = group_list_cum[i].item()
        x1_slice = x1[start:end, :]

        gmm_out = torch.matmul(x1_slice, x2)
        act, gate = gmm_out.chunk(2, dim=-1)
        swiglu_out = act / (1.0 + torch.exp(-act)) * gate
        swiglu_out = swiglu_out.to(torch.bfloat16).to(torch.float32)

        q_out, q_scale_out = _mx_fp8_quant_output(swiglu_out)
        grouped_q.append(q_out)
        grouped_q_scale.append(q_scale_out)

    expected_q = torch.cat(grouped_q, dim=0)
    expected_q_scale = torch.cat(grouped_q_scale, dim=0)

    return (a_fp8_npu, b_fp8_trans_npu, a_scale_npu, b_scale_trans_npu,
            group_list_noncum, expected_q, expected_q_scale)


def prepare_fp8_mx_grouped_matmul_finalize_routing_inputs(
    m: int,
    n: int,
    k: int,
    problem_count: int,
    batch: int,
    data_parallel_size: int = 1,
    enable_bias: bool = True,
    enable_shared_input: bool = True,
    shared_input_weight: float = 0.5,
    shared_input_offset: int = 0,
    group_list_type: int = 0,
    trans_b: bool = False,
    quant_type: torch.dtype = torch.float8_e4m3fn,
    device: str = "npu",
) -> Tuple[torch.Tensor, ...]:
    """Build MX-FP8 grouped matmul + finalize routing inputs and golden (example 65).

    Returns:
        (a_fp8, b_fp8, a_scale, b_scale, group_list, logit, row_index,
         bias, shared_input, expected)
    """
    from itertools import accumulate

    _FP8_FORMAT = {
        torch.float8_e4m3fn: (8, 448.0),
        torch.float8_e5m2: (15, 57344.0),
    }
    fp8_dtype = quant_type
    emax, fp8_max = _FP8_FORMAT[quant_type]
    bsdp = batch // data_parallel_size

    if problem_count == 1:
        group_sizes = [m]
    else:
        weights = list(range(1, problem_count + 1))
        total_weight = sum(weights)
        group_sizes = [max(1, m * w // total_weight) for w in weights]
        diff = m - sum(group_sizes)
        group_sizes[-1] += diff

    if group_list_type == 0:
        group_list = torch.tensor(list(accumulate(group_sizes)), dtype=torch.int64)
    else:
        group_list = torch.tensor(group_sizes, dtype=torch.int64)

    a_fp32 = torch.randn(m, k, dtype=torch.float32)
    a_fp8, a_scale, a_deq = _quantize_axis_last(a_fp32, fp8_dtype, emax, fp8_max)
    a_scale = a_scale.reshape(m, a_scale.shape[1] // 2, 2).contiguous()

    b_deq_list = []
    b_fp8_list = []
    b_scale_list = []
    for _ in range(problem_count):
        b_fp32 = torch.randn(k, n, dtype=torch.float32)
        b_fp8_i, b_scale_i, b_deq_i = _quantize_axis_first(b_fp32, fp8_dtype, emax, fp8_max)
        b_scale_i = b_scale_i.reshape(b_scale_i.shape[0] // 2, 2, b_scale_i.shape[1])
        if trans_b:
            b_fp8_i = b_fp8_i.t().contiguous()
            b_scale_i = b_scale_i.permute(2, 0, 1).contiguous()
        else:
            b_scale_i = b_scale_i.permute(0, 2, 1).contiguous()
        b_fp8_list.append(b_fp8_i)
        b_scale_list.append(b_scale_i)
        b_deq_list.append(b_deq_i)

    b_fp8 = torch.stack(b_fp8_list, dim=0)
    b_scale = torch.stack(b_scale_list, dim=0)

    logit = torch.randn(m, dtype=torch.float32)
    row_index = torch.arange(m, dtype=torch.int64) % batch

    bias = torch.empty(0, dtype=torch.bfloat16)
    if enable_bias:
        bias = torch.randn(problem_count, n, dtype=torch.bfloat16)

    shared_input = torch.empty(0, dtype=torch.bfloat16)
    if enable_shared_input:
        shared_input = torch.randn(bsdp, n, dtype=torch.bfloat16)

    c_parts = []
    offset = 0
    for i in range(problem_count):
        mi = group_sizes[i]
        a_slice = a_deq[offset:offset + mi, :]
        c_part = torch.matmul(a_slice, b_deq_list[i])
        if enable_bias:
            c_part += bias[i, :].to(torch.float32).unsqueeze(0)
        c_parts.append(c_part)
        offset += mi

    c_fp32 = torch.cat(c_parts, dim=0)
    out_fp32 = torch.zeros(batch, n, dtype=torch.float32)

    if enable_shared_input:
        contribution = shared_input_weight * shared_input.to(torch.float32)
        out_fp32[shared_input_offset:(shared_input_offset + bsdp), :] += contribution

    weighted = logit.unsqueeze(1) * c_fp32
    out_fp32.index_add_(0, row_index, weighted)

    if device == "npu":
        a_fp8 = a_fp8.contiguous().npu()
        b_fp8 = b_fp8.contiguous().npu()
        a_scale = a_scale.contiguous().npu()
        b_scale = b_scale.contiguous().npu()
        group_list = group_list.contiguous().npu()
        logit = logit.contiguous().npu()
        row_index = row_index.contiguous().npu()
        if enable_bias:
            bias = bias.contiguous().npu()
        if enable_shared_input:
            shared_input = shared_input.contiguous().npu()

    return a_fp8, b_fp8, a_scale, b_scale, group_list, logit, row_index, bias, shared_input, out_fp32
