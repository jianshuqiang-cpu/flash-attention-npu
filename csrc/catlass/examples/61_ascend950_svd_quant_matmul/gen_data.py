#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
from typing import Dict, Optional, Tuple

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

dtype_map = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
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


_FP4_LUT = {
    "E2M1": _build_fp4_lut("E2M1"),
    "E1M2": _build_fp4_lut("E1M2"),
}


def _quantize_to_fp4_lut(values: torch.Tensor, format_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)
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


def _pack_fp4_nibbles(index_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = index_matrix.shape
    if cols % 2 != 0:
        index_matrix = torch.cat(
            [index_matrix, torch.zeros((rows, 1), dtype=torch.uint8, device=index_matrix.device)],
            dim=1,
        )

    low = index_matrix[:, 0::2]
    high = index_matrix[:, 1::2] << 4
    packed = low | high
    return packed.to(torch.uint8)


def _quantize_axis_last(
    matrix: torch.Tensor,
    format_name: str,
    block_size: int,
    is_normal_matrix: bool = True,
    qmax: float = 1.0
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

    if is_normal_matrix:
        exp = torch.floor(torch.log2(torch.clamp(max_abs, min=_EPSILON))) - _FP4_FORMATS[format_name]["emax"]
    else:
        qmaxInv = torch.tensor(1.0 / qmax, dtype=matrix.dtype, device=matrix.device)
        exp = torch.ceil(torch.log2(torch.clamp(max_abs, min=_EPSILON) * qmaxInv))

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
        scale_padded = torch.ones((m, padded_blocks), dtype=torch.float32, device=matrix.device)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quantized, scale, dequantized


def _quantize_axis_first(
    matrix: torch.Tensor,
    format_name: str,
    block_size: int,
    is_normal_matrix: bool = True,
    qmax: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_axis_last(matrix.t().contiguous(), format_name, block_size, is_normal_matrix, qmax)
    return quantized_t.t().contiguous(), scale_t.t().contiguous(), dequantized_t.t().contiguous()


def _quantize(
    matrix: torch.Tensor,
    format_name: str,
    axis: int,
    block_size: int = _BLOCK_SIZE,
    is_normal_matrix: bool = True,
    qmax: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first(matrix, format_name, block_size, is_normal_matrix, qmax)
    if axis == 1:
        return _quantize_axis_last(matrix, format_name, block_size, is_normal_matrix, qmax)


def gen_data_matrix16_fp4_e2m1(matrix: torch.Tensor, axis: int, trans: int, is_normal_matrix = True, qmax = 1.0):
    assert matrix.dtype in [torch.float16, torch.bfloat16], f"unsupported dtype {matrix.dtype}"
    assert matrix.is_contiguous(), "matrix must be contiguous"
    assert axis in [0, 1], f"unsupported axis {axis}"
    assert trans in [0, 1], f"unsupported trans {trans}"

    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E2M1", axis, _BLOCK_SIZE, is_normal_matrix, qmax)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    _is_normal_matrix, _qmax = True, 1.0 # reset

    return quantized_matrix_uint8, scale_matrix.to(torch.float8_e8m0fnu), dequantized_matrix


def generate_svd_quant_matmul_data(args):
    m, n, k, r = args.m, args.n, args.k, args.r
    qmax = args.qmax
    dtype16 = torch.float16 if args.dtype == "float16" else "bfloat16"

    print("generating data: ", vars(args))

    trans_x, trans_w = 0, 1

    x_fp16 = torch.randn(m, k, dtype=dtype16)
    svd1 = torch.randn(k, r, dtype=dtype16)
    svd2 = torch.randn(r, n, dtype=dtype16)

    if args.smooth:
        # NPU实现是 fp16的 x @ smooth
        smooth_scale = 1.0 + torch.randn(k, dtype=torch.float32) * 0.1
        smooth_scale_inv = (1 / smooth_scale).to(dtype16) # fp16
        smooth_x = x_fp16 @ torch.diag(smooth_scale_inv) # X' = X @ diag(s^-1)
        # 高精度smoothX
        smooth_x_fp32 = x_fp16.to(torch.float32) @ torch.diag(1 / smooth_scale)
    else:
        smooth_scale_inv = None
        smooth_x = x_fp16

    if args.bias:
        bias = torch.randn(n, dtype=torch.float32)
    else:
        bias = None

    smooth_x_fp4, smooth_x_scale, deq_smooth_x_fp32 = gen_data_matrix16_fp4_e2m1(smooth_x, 1, trans_x, is_normal_matrix=False, qmax=qmax) # m*k

    # 这里 w 实际是公式中的残差 R
    w_fp16 = torch.randn(k, n, dtype=dtype16) * 0.1
    w_fp4, w_scale, deq_w_fp32 = gen_data_matrix16_fp4_e2m1(w_fp16, 0, trans_w, is_normal_matrix=True) # k*n
    w_scale = w_scale.reshape(w_scale.shape[0] // 2, 2, w_scale.shape[1])

    c1_fp32 = smooth_x.to(torch.float32) @ svd1.to(torch.float32) # mmad1
    c1_fp16 = c1_fp32.to(dtype16)

    # y_cpu 完全模拟NPU实现
    # y_golden 高精度，不考虑中间的精度损失
    if bias is None:
        y_cpu = c1_fp16.to(torch.float32) @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32
        y_golden = smooth_x_fp32 @ svd1.to(torch.float32) @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32
    else:
        y_cpu = c1_fp16.to(torch.float32) @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32 + bias
        y_golden = c1_fp32 @ svd2.to(torch.float32) + deq_smooth_x_fp32 @ deq_w_fp32 + bias

    svd1 = svd1.permute(1, 0).contiguous().permute(1, 0) # shape(k, n) 但是转置layout
    svd2 = svd2.permute(1, 0).contiguous().permute(1, 0) # shape(r, n) 但是转置layout
    w_fp4 = torch.tensor(w_fp4.contiguous().untyped_storage(), dtype=torch.int8)
    w_scale = torch.tensor(w_scale.permute(2, 0, 1).contiguous().untyped_storage(), dtype=torch.int8)

    return {
        "x": x_fp16,
        "svd1": svd1,
        "svd2": svd2,
        "w": w_fp4,
        "w_scale": w_scale,
        "qmax":qmax,

        "smooth": smooth_scale_inv,
        "bias": bias,
        "y_cpu": y_cpu,
        "y_golden": y_golden,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16", help="default float16")
    parser.add_argument("--smooth", type=int, choices=[0, 1], default=1, help="default 1")
    parser.add_argument("--bias", type=int, choices=[0, 1], default=0, help="default 0")
    parser.add_argument("--qmax", type=float, default=8.0, help="qmax>0 default 8.0")
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('r', type=int)

    args = parser.parse_args()

    data = generate_svd_quant_matmul_data(args)
    save_path = f"{_SCRIPT_DIR}/data/"
    os.system(f"rm -rf {save_path}")
    os.system(f"mkdir -p {save_path}/input {save_path}/golden")

    dtype16 = data["x"].dtype
    torch.tensor(data["x"].untyped_storage(), dtype=dtype16).numpy().tofile(f"{save_path}/input/x.bin")
    torch.tensor(data["svd1"].untyped_storage(), dtype=dtype16).numpy().tofile(f"{save_path}/input/svd1.bin")
    torch.tensor(data["svd2"].untyped_storage(), dtype=dtype16).numpy().tofile(f"{save_path}/input/svd2.bin")
    torch.tensor(data["w"].untyped_storage(), dtype=torch.int8).numpy().tofile(f"{save_path}/input/w.bin")
    torch.tensor(data["w_scale"].untyped_storage(), dtype=torch.int8).numpy().tofile(f"{save_path}/input/w_scale.bin")
    torch.tensor([data["qmax"]], dtype=torch.float32).numpy().tofile(f"{save_path}/input/qmax.bin")

    if args.smooth:
        torch.tensor(data["smooth"].untyped_storage(), dtype=dtype16).numpy().tofile(f"{save_path}/input/smooth_scale.bin")
    if args.bias:
        torch.tensor(data["bias"].untyped_storage(), dtype=torch.float32).numpy().tofile(f"{save_path}/input/bias.bin")

    torch.tensor(data["y_cpu"].untyped_storage(), dtype=torch.float32).numpy().tofile(f"{save_path}/golden/y_cpu.bin")
    torch.tensor(data["y_golden"].untyped_storage(), dtype=torch.float32).numpy().tofile(f"{save_path}/golden/y_golden.bin")
    print(f"save data in {save_path}")
