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

import os
import argparse
import math
import time
import subprocess
from itertools import accumulate
from typing import Tuple, Dict

import torch
import numpy as np


##################### FP8 Quantizer
class MXFP8MatrixQuantizer:
    """
    二维矩阵 MXFP8 量化器
    支持:
    1. 自定义量化轴 (0: 行方向, 1: 列方向)
    2. 自定义分块大小
    3. FP8_E8M0FNU 缩放因子
    4. 输出量化矩阵和缩放矩阵
    """

    # MXFP8 数据格式定义
    DATA_FORMATS = {
        "E4M3": {
            "exp_bits": 4,
            "mantissa_bits": 3,
            "bias": 7,
            "emax": 8,
            "max_value": 448.0,
            "min_value": -448.0,
        },
        "E5M2": {
            "exp_bits": 5,
            "mantissa_bits": 2,
            "bias": 15,
            "emax": 15,
            "max_value": 57344.0,
            "min_value": -57344.0,
        },
    }

    # FP8_E8M0FNU 缩放格式定义
    SCALE_FORMAT = {
        "name": "FP8_E8M0FNU",
        "exp_bits": 8,
        "mantissa_bits": 0,
        "bias": 128,  # 偏置为128，0表示指数-128
        "max_exp": 127,
        "min_exp": -128,
        "max_value": 2**127,  # 约 1.7e38
        "min_value": 2**-128,  # 约 2.9e-39
        "signed": False,  # 无符号，仅正数
        "allow_zero": True,  # 允许零值（指数-128表示零）
    }

    def __init__(
        self,
        data_format: str = "E4M3",
        axis: int = 1,
        block_size: int = 32,
        epsilon: float = 1e-12,
    ):
        """
        初始化 MXFP8 矩阵量化器

        参数:
            data_format: 数据格式 'E4M3' 或 'E5M2'
            axis: 量化轴 (0: 行方向, 1: 列方向)
            block_size: 分块大小 (在量化轴上的元素数)
            epsilon: 防止除零的小值
        """
        if data_format not in self.DATA_FORMATS:
            raise ValueError(
                f"不支持的数据格式: {data_format}，支持: {list(self.DATA_FORMATS.keys())}"
            )

        if axis not in [0, 1]:
            raise ValueError("axis 必须是 0 (行) 或 1 (列)")

        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")

        self.data_format = data_format
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        self.config = self.DATA_FORMATS[data_format]

        # 预构建 FP8 值查找表
        self._build_fp8_lookup_table()

    def _build_fp8_lookup_table(self):
        """构建 FP8 值查找表"""
        if self.data_format == "E4M3":
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()

    def _build_e4m3_lookup_table(self):
        """构建 E4M3 值查找表"""
        values = []

        # E4M3: 总共 256 个可能值 (0-255)
        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:  # NaN，替换为最大值
                value = sign * self.config["max_value"]
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config["bias"]))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 8.0) * (
                        2.0 ** (exp - self.config["bias"])
                    )

                value = sign * value

            # 钳位到有效范围
            if value > self.config["max_value"]:
                value = self.config["max_value"]
            elif value < self.config["min_value"]:
                value = self.config["min_value"]

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config["min_value"]
        self.fp8_max = self.config["max_value"]

    def _build_e5m2_lookup_table(self):
        """构建 E5M2 值查找表"""
        values = []

        for i in range(256):
            # 解码
            if i < 128:  # 正数
                sign = 1
                val = i
            else:  # 负数
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:  # NaN/Inf，替换为最大值
                value = sign * self.config["max_value"]
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config["bias"]))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 4.0) * (
                        2.0 ** (exp - self.config["bias"])
                    )

                value = sign * value

            # 钳位到有效范围
            if value > self.config["max_value"]:
                value = self.config["max_value"]
            elif value < self.config["min_value"]:
                value = self.config["min_value"]

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config["min_value"]
        self.fp8_max = self.config["max_value"]

    def _compute_scale_vectorized(self, max_abs: torch.Tensor) -> torch.Tensor:
        """
        计算 FP8_E8M0FNU 格式的缩放因子

        返回:
            scale: 缩放因子 (浮点数)
            exp: 指数 (E8M0FNU 格式)
        """
        zero_mask = max_abs < self.epsilon
        safe_max = torch.clamp(max_abs, min=self.epsilon)
        log2_scale = torch.log2(safe_max)
        exp = torch.floor(log2_scale).to(torch.int32) - self.config["emax"]
        exp = torch.clamp(
            exp, self.SCALE_FORMAT["min_exp"], self.SCALE_FORMAT["max_exp"]
        )
        scale = torch.pow(2.0, exp.to(torch.float32))
        scale = torch.where(zero_mask, torch.ones_like(scale), scale)
        return scale

    def _quantize_to_fp8_vectorized(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP8 格式

        使用查找表进行最近邻量化
        """
        data_clamped = torch.clamp(data, self.fp8_min, self.fp8_max)
        dist = torch.abs(data_clamped.unsqueeze(-1) - self.fp8_lut)
        indices = torch.argmin(dist, dim=-1)
        return self.fp8_lut[indices]

    def quantize_matrix(
        self, matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量化二维矩阵

        参数:
            matrix: 输入二维矩阵 [M, N]

        返回:
            quantized_matrix: 量化后的矩阵 [M, N]
            scale_matrix: 缩放因子矩阵 (或编码后的缩放因子矩阵)
        """
        if matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {matrix.dim()}")

        M, N = matrix.shape

        # 根据量化轴确定分块方式
        if self.axis == 0:  # 行方向量化
            return self._quantize_by_rows(matrix, M, N)
        else:  # 列方向量化
            return self._quantize_by_cols(matrix, M, N)

    def _quantize_by_rows(
        self, matrix: torch.Tensor, M: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.block_size
        num_blocks = (M + B - 1) // B
        pad_rows = num_blocks * B - M
        if pad_rows > 0:
            padded = torch.nn.functional.pad(matrix, (0, 0, 0, pad_rows), value=0)
        else:
            padded = matrix
        block_view = padded.view(num_blocks, B, N)
        max_abs = torch.max(torch.abs(block_view), dim=1).values
        scales = self._compute_scale_vectorized(max_abs)
        scaled = block_view / scales.unsqueeze(1)
        quantized_blocks = self._quantize_to_fp8_vectorized(scaled)
        quantized_matrix = quantized_blocks.view(-1, N)[:M, :]
        target_h = ((num_blocks + 1) // 2) * 2
        scale_matrix = torch.ones(
            (target_h, N), dtype=torch.float32, device=matrix.device
        )
        scale_matrix[:num_blocks, :] = scales
        return quantized_matrix, scale_matrix

    def _quantize_by_cols(
        self, matrix: torch.Tensor, M: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.block_size
        num_blocks = (N + B - 1) // B
        pad_cols = num_blocks * B - N
        if pad_cols > 0:
            padded = torch.nn.functional.pad(matrix, (0, pad_cols, 0, 0), value=0)
        else:
            padded = matrix
        block_view = padded.view(M, num_blocks, B)
        max_abs = torch.max(torch.abs(block_view), dim=2).values
        scales = self._compute_scale_vectorized(max_abs)
        scaled = block_view / scales.unsqueeze(2)
        quantized_blocks = self._quantize_to_fp8_vectorized(scaled)
        quantized_matrix = quantized_blocks.view(M, -1)[:, :N]
        target_w = ((num_blocks + 1) // 2) * 2
        scale_matrix = torch.ones(
            (M, target_w), dtype=torch.float32, device=matrix.device
        )
        scale_matrix[:, :num_blocks] = scales
        return quantized_matrix, scale_matrix

    def dequantize_matrix(
        self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        反量化矩阵

        参数:
            quantized_matrix: 量化后的矩阵
            scale_matrix: 缩放因子矩阵（或编码矩阵）

        返回:
            dequantized_matrix: 反量化后的矩阵
        """
        M, N = quantized_matrix.shape

        # 根据量化轴进行反量化
        if self.axis == 0:  # 行方向
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:  # 列方向
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_rows(
        self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int
    ) -> torch.Tensor:
        """行方向反量化"""
        scales_expanded = scale_matrix.repeat_interleave(self.block_size, dim=0)[:M, :]
        scales_expanded = torch.where(
            torch.abs(scales_expanded) > self.epsilon,
            scales_expanded,
            torch.ones_like(scales_expanded),
        )
        return quantized_matrix * scales_expanded

    def _dequantize_by_cols(
        self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int
    ) -> torch.Tensor:
        """列方向反量化"""
        scales_expanded = scale_matrix.repeat_interleave(self.block_size, dim=1)[:, :N]
        scales_expanded = torch.where(
            torch.abs(scales_expanded) > self.epsilon,
            scales_expanded,
            torch.ones_like(scales_expanded),
        )
        return quantized_matrix * scales_expanded


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10

    quantizer = MXFP8MatrixQuantizer(data_format="E4M3", axis=axis, block_size=32)

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix, matrix


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)

    quantizer = MXFP8MatrixQuantizer(data_format="E5M2", axis=axis, block_size=32)

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix, matrix


################# FP4 Quantizer

_BLOCK_SIZE = 32
_EPSILON = 1e-12
_MIN_SCALE_EXP = -128
_MAX_SCALE_EXP = 127

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


def _quantize_to_fp4_lut(
    values: torch.Tensor, format_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    lut = _FP4_LUT[format_name].to(values.device)
    min_value = _FP4_FORMATS[format_name]["min_value"]
    max_value = _FP4_FORMATS[format_name]["max_value"]

    clamped = values.clamp(min_value, max_value)

    # 与原实现一致：按 LUT 顺序 argmin，距离相等时取更小下标。
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


def _quantize_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int
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


def _quantize_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_t, scale_t, dequantized_t = _quantize_axis_last(
        matrix.t().contiguous(), format_name, block_size
    )
    return (
        quantized_t.t().contiguous(),
        scale_t.t().contiguous(),
        dequantized_t.t().contiguous(),
    )


def _quantize(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp4_e2m1(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E2M1", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E2M1")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    dequantized_fp8 = dequantized_matrix.to(torch.float8_e8m0fnu)
    return (
        quantized_matrix_uint8,
        scale_matrix.to(torch.float8_e8m0fnu),
        dequantized_matrix,
        dequantized_fp8,
    )


def gen_data_fp4_e1m2(row, col, axis, trans):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantized_matrix, scale_matrix, dequantized_matrix = _quantize(matrix, "E1M2", axis)

    if trans == 1:
        quantized_matrix = quantized_matrix.t().contiguous()

    _, fp4_indices = _quantize_to_fp4_lut(quantized_matrix, "E1M2")
    quantized_matrix_uint8 = _pack_fp4_nibbles(fp4_indices)

    dequantized_fp8 = dequantized_matrix.to(torch.float8_e8m0fnu)
    return (
        quantized_matrix_uint8,
        scale_matrix.to(torch.float8_e8m0fnu),
        dequantized_matrix,
        dequantized_fp8,
    )


##################### FP8 Data generator (dequantized, scale, golden and upgraded(original) )
def gen_data_fp8(
    m, n, k, trans_a, trans_b, problem_count, quant_type=torch.float8_e5m2
):
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./golden", exist_ok=True)

    if quant_type == torch.float8_e4m3fn:
        a_fp8, a_scale, a_fp32, a_upgrade = gen_data_fp8_e4m3(m, k, 1)
        b_fp8_i, b_scale_i, b_fp32_i, b_upgrade_i = gen_data_fp8_e4m3(k, n, 0)
    elif quant_type == torch.float8_e5m2:
        a_fp8, a_scale, a_fp32, a_upgrade = gen_data_fp8_e5m2(m, k, 1)
        b_fp8_i, b_scale_i, b_fp32_i, b_upgrade_i = gen_data_fp8_e5m2(k, n, 0)
    else:
        raise ValueError(
            f"不支持的数据格式: {quant_type}，支持: {[torch.float8_e4m3fn, torch.float8_e5m2]}"
        )

    # convert scale to mx type
    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
    b_scale_i = b_scale_i.reshape(b_scale_i.shape[0] // 2, 2, b_scale_i.shape[1])

    if trans_a:
        a_fp8 = a_fp8.t()
        a_scale = a_scale.permute(1, 0, 2)  # [k//64, m, 2]

    if trans_b:
        b_fp8_i = b_fp8_i.t().contiguous()
        b_scale_i = b_scale_i.permute(2, 0, 1)  # [n, k//64, 2]
    else:
        b_scale_i = b_scale_i.permute(0, 2, 1)  # [k//64, n, 2]

    # Generate the total weight matrix (and relative scale factor)
    if trans_b:
        b_fp8 = torch.zeros((problem_count, n, k), dtype=quant_type)
        b_scale = torch.zeros(
            (problem_count, n, math.ceil(k / 64), 2), dtype=torch.float8_e8m0fnu
        )
    else:
        b_fp8 = torch.zeros((problem_count, k, n), dtype=quant_type)
        b_scale = torch.zeros(
            (problem_count, math.ceil(k / 64), n, 2), dtype=torch.float8_e8m0fnu
        )
    b_fp32 = torch.zeros((problem_count, k, n), dtype=torch.float32)
    b_upgrade = torch.zeros((problem_count, k, n), dtype=torch.float32)
    for i in range(problem_count):
        # Identity weight for each group
        b_fp8[i, :, :] = b_fp8_i
        b_scale[i, :, :, :] = b_scale_i
        b_fp32[i, :, :] = b_fp32_i
        b_upgrade[i, :, :] = b_upgrade_i

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile("./data/a_8.bin")
    b_np.tofile("./data/b_8.bin")

    a_scale_np = torch.tensor(
        a_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_scale_np.tofile("./data/a_scale.bin")
    b_scale_np.tofile("./data/b_scale.bin")

    # caculated golden
    a_reshaped = a_fp32.reshape(problem_count, m // problem_count, k)
    c_3d = torch.matmul(a_reshaped, b_fp32)  # [g, m_i, k] @ [g, k, n]
    c_fp32 = c_3d.view(-1, n)

    # caculate upgraded (higher precison)
    aup_reshaped = a_upgrade.reshape(problem_count, m // problem_count, k)
    c_3d_upgrade = aup_reshaped.to(torch.float32) @ b_upgrade.to(torch.float32)
    c_upgrade = c_3d_upgrade.view(-1, n)

    group_list = []
    for i in range(problem_count):
        group_list.append(m // problem_count)
    cumsum_list = list(accumulate(group_list))

    groups = problem_count
    data_fp32 = np.zeros((m, n)).astype(np.float32)
    data_upgrade = np.zeros((m, n)).astype(np.float32)
    start_row = 0
    for group_id in range(groups):
        end_row = cumsum_list[group_id]
        data_fp32[start_row:end_row, :] = c_fp32[start_row:end_row, :]
        data_upgrade[start_row:end_row, :] = c_upgrade[start_row:end_row, :]
        start_row = end_row

    return data_fp32, data_upgrade


##################### FP4 Data generator
def gen_data_fp4(m, n, k, trans_a, trans_b, problem_count, quant_type="e2m1"):
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./golden", exist_ok=True)

    if quant_type == "e2m1":
        a_uint8, a_scale, a_fp32, a_fp8 = gen_data_fp4_e2m1(m, k, 1, trans_a)
        b_uint8_i, b_scale_i, b_fp32_i, b_fp8_i = gen_data_fp4_e2m1(k, n, 0, trans_b)
    elif quant_type == "e1m2":
        a_uint8, a_scale, a_fp32, a_fp8 = gen_data_fp4_e1m2(m, k, 1, trans_a)
        b_uint8_i, b_scale_i, b_fp32_i, b_fp8_i = gen_data_fp4_e1m2(k, n, 0, trans_b)
    else:
        raise ValueError(f"不支持的数据格式: {quant_type}，FP4支持: {['e2m1', 'e1m2']}")

    b_scale_i = b_scale_i.reshape(b_scale_i.shape[0] // 2, 2, b_scale_i.shape[1])

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)

    if trans_a:
        a_scale = a_scale.permute(1, 0, 2)

    if trans_b:
        b_scale_i = b_scale_i.permute(2, 0, 1)
    else:
        b_scale_i = b_scale_i.permute(0, 2, 1)

    if trans_b:
        b_uint8 = torch.zeros((problem_count, n, (k + 1) // 2), dtype=torch.uint8)
        b_scale = torch.zeros(
            (problem_count, n, math.ceil(k / 64), 2), dtype=torch.float8_e8m0fnu
        )
    else:
        b_uint8 = torch.zeros((problem_count, k, (n + 1) // 2), dtype=torch.uint8)
        b_scale = torch.zeros(
            (problem_count, math.ceil(k / 64), n, 2), dtype=torch.float8_e8m0fnu
        )
    b_fp32 = torch.zeros((problem_count, k, n), dtype=torch.float32)
    b_fp8 = torch.zeros((problem_count, k, n), dtype=torch.float32)
    for i in range(problem_count):
        # Identity weight for each group
        b_uint8[i, :, :] = b_uint8_i
        b_scale[i, :, :, :] = b_scale_i
        b_fp32[i, :, :] = b_fp32_i
        b_fp8[i, :, :] = b_fp8_i

    a_np = torch.tensor(a_uint8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_uint8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile("./data/a_8.bin")
    b_np.tofile("./data/b_8.bin")

    a_scale_np = torch.tensor(
        a_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_scale_np.tofile("./data/a_scale.bin")
    b_scale_np.tofile("./data/b_scale.bin")

    a_reshaped = a_fp32.reshape(problem_count, m // problem_count, k)
    c_3d = a_reshaped @ b_fp32
    c_fp32 = c_3d.view(-1, n)

    afp8_reshaped = a_fp8.reshape(problem_count, m // problem_count, k)
    c_3d_fp8 = afp8_reshaped.to(torch.float32) @ b_fp8.to(torch.float32)
    c_fp8 = c_3d_fp8.view(-1, n)

    group_list = []
    for i in range(problem_count):
        group_list.append(m // problem_count)
    cumsum_list = list(accumulate(group_list))

    groups = problem_count
    data_fp32 = np.zeros((m, n)).astype(np.float32)
    data_fp8 = np.zeros((m, n)).astype(np.float32)
    start_row = 0
    for group_id in range(groups):
        end_row = cumsum_list[group_id]
        data_fp32[start_row:end_row, :] = c_fp32[start_row:end_row, :]
        data_fp8[start_row:end_row, :] = c_fp8[start_row:end_row, :]
        start_row = end_row

    return data_fp32, data_fp8


##################### Double-standard evalucator
def get_error_floor(dtype):
    # This follows the device output precision, not result.bin's FP32 storage type.
    if dtype == "half":
        return 2 ** (-11)
    if dtype == "bf16":
        return 2 ** (-8)
    return 2 ** (-12)


def compute_rela_errors(result, golden):
    relative_errors = np.abs((result - golden) / (golden + 1e-7))
    mse = np.mean((result - golden) ** 2)
    return (
        np.max(relative_errors),
        np.mean(relative_errors),
        np.sqrt(mse),
    )  # 分别计算MARE、MERE、RMSE


def compare_mare(mare_npu, mare_upgrade, dtype):
    err = get_error_floor(dtype)
    res = True if (mare_npu / max(mare_upgrade, err)) < 10 else False
    return res


def compare_mere(mere_npu, mere_upgrade, dtype):
    err = get_error_floor(dtype)
    res = True if (mere_npu / max(mere_upgrade, err)) < 2 else False
    return res


def compare_rmse(rmse_npu, rmse_upgrade, dtype):
    err = get_error_floor(dtype)
    res = True if (rmse_npu / max(rmse_upgrade, err)) < 2 else False
    return res


if __name__ == "__main__":
    dtype = "bf16"
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_count", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_b", type=int)
    parser.add_argument("quant_type", type=str, default="float8_e5m2")
    parser.add_argument("device_id", type=int)
    # 可选：选择 output/bin 下的可执行文件名。默认基线；ASWT+tail 变体传
    # --bin 55_ascend950_mx_grouped_matmul_slice_m_aswt 即可用同一脚本做精度比对。
    default_bin_name = os.environ.get(
        "CATLASS_EXAMPLE_BIN_NAME", "55_ascend950_mx_grouped_matmul_slice_m"
    )
    parser.add_argument("--bin", type=str, default=default_bin_name, dest="bin_name")
    args = parser.parse_args()
    m = args.m
    n = args.n
    k = args.k
    problem_count = args.problem_count
    trans_b = args.trans_b
    quant_type = args.quant_type
    device_id = args.device_id

    print("------计算golden------")

    if quant_type == "float8_e5m2":
        golden_data, golden_upgrade = gen_data_fp8(
            args.m, args.n, args.k, 0, trans_b, problem_count, torch.float8_e5m2
        )
    elif quant_type == "float8_e4m3fn":
        golden_data, golden_upgrade = gen_data_fp8(
            args.m, args.n, args.k, 0, trans_b, problem_count, torch.float8_e4m3fn
        )
    elif quant_type == "float4_e2m1fn_x2":
        golden_data, golden_upgrade = gen_data_fp4(
            args.m, args.n, args.k, 0, trans_b, problem_count, "e2m1"
        )
    else:
        raise ValueError(
            f"不支持的数据格式: {quant_type}，支持: "
            f"{['float8_e4m3fn', 'float8_e5m2', 'float4_e2m1fn_x2']}"
        )

    # 获取算子路径并执行算子用例
    current_dir = os.path.dirname(os.path.abspath(__file__))
    catlass_home_dir = os.path.dirname(os.path.dirname(current_dir))
    op_path = os.path.join(catlass_home_dir, "output", "bin", args.bin_name)
    print(f"npu op path = {op_path}")

    result = subprocess.run(
        [
            op_path,
            str(problem_count),
            str(m),
            str(n),
            str(k),
            str(trans_b),
            quant_type,
            str(device_id),
        ],
        capture_output=True,
        text=True,
    )
    print(f"npu op run log = {result.stdout}")

    print("------ 计算相对误差 -----")
    result_data = np.fromfile("./data/result.bin", dtype=np.float32).reshape(m, n)
    result_data = result_data.flatten()  # npu计算结果
    golden_data = golden_data.flatten()  # 同精度计算结果
    golden_upgrade = golden_upgrade.flatten()  # 升精度的计算结果，当做真值
    mare_npu, mere_npu, rmse_npu = compute_rela_errors(result_data, golden_upgrade)
    mare_upgrade, mere_upgrade, rmse_upgrade = compute_rela_errors(
        golden_data, golden_upgrade
    )

    print("------ 综合精度指标 ------")
    print(f"npu mare = {mare_npu:.4f}, upgrade mare = {mare_upgrade:.6f}")
    print(f"npu mere = {mere_npu:.4f}, upgrade mere = {mere_upgrade:.6f}")
    print(f"npu rmse = {rmse_npu:.4f}, upgrade rmse = {rmse_upgrade:.6f}")

    print("------ 开始比较 ------")
    mare_info = compare_mare(mare_npu, mare_upgrade, dtype)
    mere_info = compare_mere(mere_npu, mere_upgrade, dtype)
    rmse_info = compare_rmse(rmse_npu, rmse_upgrade, dtype)
    res = "Compare success" if (mare_info & mere_info & rmse_info) else "Compare false"
    print(f"比较结果：{res}")
