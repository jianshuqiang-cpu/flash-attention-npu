#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import os
from typing import Dict, Tuple

import torch

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

_FP8_FORMATS = {
    "E4M3": {
        "torch_dtype": torch.float8_e4m3fn,
        "exp_bits": 4,
        "mantissa_bits": 3,
        "bias": 7,
        "emax": 8,
        "max_value": 448.0,
        "min_value": -448.0,
    },
    "E5M2": {
        "torch_dtype": torch.float8_e5m2,
        "exp_bits": 5,
        "mantissa_bits": 2,
        "bias": 15,
        "emax": 15,
        "max_value": 57344.0,
        "min_value": -57344.0,
    },
}

_FP4_FORMATS: Dict[str, Dict[str, float]] = {
    "E2M1": {
        "exp_bits": 2,
        "mantissa_bits": 1,
        "bias": 1,
        "emax": 2,
        "max_value": 6.0,
        "min_value": -6.0,
    },
    "E1M2": {
        "exp_bits": 1,
        "mantissa_bits": 2,
        "bias": 1,
        "emax": 0,
        "max_value": 1.75,
        "min_value": -1.75,
    },
}


def _build_e4m3_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E4M3"]["bias"]
    fp8_max = _FP8_FORMATS["E4M3"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif val == 127:
            v = sign * fp8_max
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                v = (mantissa / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 8.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


def _build_e5m2_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E5M2"]["bias"]
    fp8_max = _FP8_FORMATS["E5M2"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif 124 <= val <= 127:
            v = sign * fp8_max
        else:
            exp = (val >> 2) & 0x1F
            mantissa = val & 0x03
            if exp == 0:
                v = (mantissa / 4.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 4.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


_FP8_LUT_BUILDERS = {"E4M3": _build_e4m3_lut, "E5M2": _build_e5m2_lut}
_FP8_LUT_CACHE = {}
_FP8_LUT_POS_CACHE = {}


def _get_fp8_lut(format_name: str) -> torch.Tensor:
    if format_name not in _FP8_LUT_CACHE:
        _FP8_LUT_CACHE[format_name] = _FP8_LUT_BUILDERS[format_name]()
    return _FP8_LUT_CACHE[format_name]


def _get_fp8_lut_pos(format_name: str) -> torch.Tensor:
    if format_name not in _FP8_LUT_POS_CACHE:
        full = _get_fp8_lut(format_name)
        pos = full[:128].contiguous()
        diffs = pos[1:] - pos[:-1]
        if (diffs < 0).any():
            raise AssertionError(
                f"{format_name} positive LUT half is not non-decreasing"
            )
        _FP8_LUT_POS_CACHE[format_name] = pos
    return _FP8_LUT_POS_CACHE[format_name]


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
            value = (1.0 + mantissa / float(1 << mantissa_bits)) * (
                2.0 ** (float(exp) - bias)
            )

        if sign == 1:
            value = -value
        values.append(value)

    return torch.tensor(values, dtype=torch.float32)


_FP4_LUT = {
    "E2M1": _build_fp4_lut("E2M1"),
    "E1M2": _build_fp4_lut("E1M2"),
}


def _e8m0_exp(
    max_abs: torch.Tensor, emax: int, epsilon: float = _EPSILON
) -> torch.Tensor:
    assert max_abs.dtype == torch.float32, max_abs.dtype
    zero_mask = max_abs < epsilon
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    bits = safe.contiguous().view(torch.int32)
    exp_bits = (bits >> 23) & 0xFF
    exp = exp_bits - 127 - emax
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


def _vectorized_lut_quantize_fp8(
    scaled: torch.Tensor, format_name: str, fp8_dtype: torch.dtype
) -> torch.Tensor:
    lut_pos = _get_fp8_lut_pos(format_name)
    last_idx = lut_pos.numel() - 1

    sign = torch.sign(scaled)
    mag = scaled.abs()

    upper_idx = torch.searchsorted(lut_pos, mag).clamp(max=last_idx)
    lower_idx = (upper_idx - 1).clamp(min=0)

    upper_val = lut_pos[upper_idx]
    lower_val = lut_pos[lower_idx]

    pick_lower = (mag - lower_val) <= (upper_val - mag)
    chosen_mag = torch.where(pick_lower, lower_val, upper_val)

    snapped_fp32 = sign * chosen_mag

    zero_mask = chosen_mag == 0
    snapped_fp32 = torch.where(zero_mask, torch.zeros_like(snapped_fp32), snapped_fp32)

    return snapped_fp32.to(fp8_dtype)


def _quantize_to_fp4_lut(
    values: torch.Tensor, format_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)

    distances = (clamped.unsqueeze(-1) - lut).abs()
    indices = torch.argmin(distances, dim=-1)
    quantized = lut[indices]

    return quantized, indices.to(torch.uint8)


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [
                index_matrix,
                torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device),
            ],
            dim=1,
        )

    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    packed = low | high
    return packed.to(torch.uint8)


def _quantize_fp8_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = matrix.shape
    fmt = _FP8_FORMATS[format_name]
    fp8_dtype = fmt["torch_dtype"]
    fp8_emax = fmt["emax"]
    fp8_max = fmt["max_value"]

    num_blocks = (N + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != N:
        padded = torch.zeros(M, padded_n, dtype=matrix.dtype)
        padded[:, :N] = matrix
    else:
        padded = matrix

    blocks = padded.view(M, num_blocks, block_size)

    max_abs = blocks.abs().amax(dim=-1)
    exp = _e8m0_exp(max_abs, fp8_emax)
    scale = torch.exp2(exp.to(torch.float32))

    scaled = blocks / scale.unsqueeze(-1)
    scaled_clamped = scaled.clamp(-fp8_max, fp8_max)

    quant_fp8 = _vectorized_lut_quantize_fp8(scaled_clamped, format_name, fp8_dtype)

    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    if padded_n != N:
        quant_fp8 = quant_fp8.reshape(M, padded_n)[:, :N].contiguous()
        dequant = dequant.reshape(M, padded_n)[:, :N].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(M, N)
        dequant = dequant.reshape(M, N)

    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((M, padded_blocks), dtype=torch.float32)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale, dequant


def _quantize_fp8_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qt, st, dt = _quantize_fp8_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return (qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous())


def _quantize_fp8(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp8_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp8_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def _quantize_fp4_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
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

    exp = (
        torch.floor(torch.log2(torch.clamp(max_abs, min=_EPSILON)))
        - _FP4_FORMATS[format_name]["emax"]
    )
    exp = torch.where(max_abs < _EPSILON, torch.zeros_like(exp), exp)
    exp = exp.clamp(_MIN_SCALE_EXP, _MAX_SCALE_EXP)
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
        scale_padded = torch.ones(
            (m, padded_blocks), dtype=torch.float32, device=matrix.device
        )
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_fp4_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_fp4_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return (
        quantized_t.t().contiguous(),
        scale_t.t().contiguous(),
        dequantized_t.t().contiguous(),
    )


def _quantize_fp4(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_fp4_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_fp4_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E4M3", axis)
    return (
        quant_fp8.to(torch.float8_e4m3fn),
        scale_fp32.to(torch.float8_e8m0fnu),
        dequant_fp32,
    )


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quant_fp8, scale_fp32, dequant_fp32 = _quantize_fp8(matrix, "E5M2", axis)
    return (
        quant_fp8.to(torch.float8_e5m2),
        scale_fp32.to(torch.float8_e8m0fnu),
        dequant_fp32,
    )


def gen_data_fp4_e2m1(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize_fp4(
        matrix, "E2M1", axis
    )

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    return (
        quantized_matrix_uint8,
        scale_matrix.to(torch.float8_e8m0fnu),
        dequantized_matrix,
    )


def gen_data(m, n, k, trans_a, trans_b) -> None:
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    a_fp8, a_scale, a_fp32 = gen_data_fp8_e4m3(m, k, 1)
    b_fp4, b_scale, b_fp32 = gen_data_fp4_e2m1(k, n, 0, trans_b)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

    if trans_a == 1:
        a_scale = a_scale.permute(1, 0, 2)

    if trans_b == 1:
        b_scale = b_scale.permute(2, 0, 1)
    else:
        b_scale = b_scale.permute(0, 2, 1)

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_fp4.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile(os.path.join(input_dir, "a_8.bin"))
    b_np.tofile(os.path.join(input_dir, "b_4.bin"))

    a_scale_np = torch.tensor(
        a_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    c_fp32 = a_fp32 @ b_fp32
    c_np = c_fp32.numpy()
    c_np.tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_a", type=int)
    parser.add_argument("trans_b", type=int)
    args = parser.parse_args()
    gen_data(args.m, args.n, args.k, args.trans_a, args.trans_b)
