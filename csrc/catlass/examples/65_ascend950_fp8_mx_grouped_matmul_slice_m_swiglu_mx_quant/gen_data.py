#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import argparse
import torch
import math
from typing import Tuple
import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

class MXFP8MatrixQuantizer:
    DATA_FORMATS = {
        'E4M3': {
            'exp_bits': 4,
            'mantissa_bits': 3,
            'bias': 7,
            'emax': 8,
            'max_value': 448.0,
            'min_value': -448.0
        },
        'E5M2': {
            'exp_bits': 5,
            'mantissa_bits': 2,
            'bias': 15,
            'emax': 15,
            'max_value': 57344.0,
            'min_value': -57344.0
        }
    }

    SCALE_FORMAT = {
        'name': 'FP8_E8M0FNU',
        'exp_bits': 8,
        'mantissa_bits': 0,
        'bias': 128,
        'max_exp': 127,
        'min_exp': -128,
        'max_value': 2**127,
        'min_value': 2**-128,
        'signed': False,
        'allow_zero': True
    }

    def __init__(self,
                 data_format: str = 'E4M3',
                 axis: int = 1,
                 block_size: int = 32,
                 epsilon: float = 1e-12):
        if data_format not in self.DATA_FORMATS:
            raise ValueError(f"不支持的数据格式: {data_format}")
        if axis not in [0, 1]:
            raise ValueError("axis 必须是 0 或 1")
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")

        self.data_format = data_format
        self.axis = axis
        self.block_size = block_size
        self.epsilon = epsilon
        self.config = self.DATA_FORMATS[data_format]
        self._build_fp8_lookup_table()

    def _build_fp8_lookup_table(self):
        if self.data_format == 'E4M3':
            self._build_e4m3_lookup_table()
        else:
            self._build_e5m2_lookup_table()

    def _build_e4m3_lookup_table(self):
        values = []
        for i in range(256):
            if i < 128:
                sign = 1
                val = i
            else:
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val == 127:
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07
                if exp == 0:
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))
                value = sign * value

            value = max(self.config['min_value'], min(value, self.config['max_value']))
            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _build_e5m2_lookup_table(self):
        values = []
        for i in range(256):
            if i < 128:
                sign = 1
                val = i
            else:
                sign = -1
                val = i - 128

            if val == 0:
                value = 0.0
            elif val >= 124 and val <= 127:
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03
                if exp == 0:
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))
                value = sign * value

            value = max(self.config['min_value'], min(value, self.config['max_value']))
            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> Tuple[float, int]:
        max_abs = torch.max(torch.abs(block_data)).item()
        if max_abs < self.epsilon:
            return 1.0, 0
        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128
        exp = int(math.floor(log2_scale)) - self.config['emax']
        exp = max(self.SCALE_FORMAT['min_exp'], min(exp, self.SCALE_FORMAT['max_exp']))
        scale = 2.0 ** exp
        return scale, exp

    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        original_shape = data.shape
        data_flat = data.flatten()
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)
        distances = torch.abs(data_clamped.unsqueeze(1) - self.fp8_lut.unsqueeze(0))
        min_idx = torch.argmin(distances, dim=1)
        quantized_flat = self.fp8_lut[min_idx]
        return quantized_flat.view(original_shape)

    def _process_block(self, block_data: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)
        if abs(scale) > self.epsilon:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data
        quantized_scaled = self._quantize_to_fp8(scaled_data)
        return quantized_scaled, scale, exp

    def quantize_matrix(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if matrix.dim() != 2:
            raise ValueError(f"输入必须是二维矩阵，当前维度: {matrix.dim()}")
        M, N = matrix.shape
        if self.axis == 0:
            return self._quantize_by_rows(matrix, M, N)
        else:
            return self._quantize_by_cols(matrix, M, N)

    def _quantize_by_rows(self, matrix: torch.Tensor, M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = (M + self.block_size - 1) // self.block_size
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)
            block_data = matrix[start_row:end_row, :]
            for col in range(N):
                col_data = block_data[:, col]
                quantized_col, scale, exp = self._process_block(col_data)
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def _quantize_by_cols(self, matrix: torch.Tensor, M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = (N + self.block_size - 1) // self.block_size
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            block_data = matrix[:, start_col:end_col]
            for row in range(M):
                row_data = block_data[row, :]
                quantized_row, scale, exp = self._process_block(row_data)
                quantized_matrix[row, start_col:end_col] = quantized_row
                scale_matrix[row, block_idx] = scale

        return quantized_matrix, scale_matrix

    def dequantize_matrix(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
        M, N = quantized_matrix.shape
        if self.axis == 0:
            return self._dequantize_by_rows(quantized_matrix, scale_matrix, M, N)
        else:
            return self._dequantize_by_cols(quantized_matrix, scale_matrix, M, N)

    def _dequantize_by_rows(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        num_blocks = scale_matrix.shape[0]
        dequantized_matrix = torch.zeros_like(quantized_matrix)
        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)
            for col in range(N):
                scale = scale_matrix[block_idx, col].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[start_row:end_row, col] = quantized_matrix[start_row:end_row, col] * scale
                else:
                    dequantized_matrix[start_row:end_row, col] = quantized_matrix[start_row:end_row, col]
        return dequantized_matrix

    def _dequantize_by_cols(self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int) -> torch.Tensor:
        num_blocks = scale_matrix.shape[1]
        dequantized_matrix = torch.zeros_like(quantized_matrix)
        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)
            for row in range(M):
                scale = scale_matrix[row, block_idx].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[row, start_col:end_col] = quantized_matrix[row, start_col:end_col] * scale
                else:
                    dequantized_matrix[row, start_col:end_col] = quantized_matrix[row, start_col:end_col]
        return dequantized_matrix


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10
    quantizer = MXFP8MatrixQuantizer(data_format='E4M3', axis=axis, block_size=32)
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)
    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)
    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quantizer = MXFP8MatrixQuantizer(data_format='E5M2', axis=axis, block_size=32)
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)
    quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)
    return quantized_matrix, scale_matrix, dequantized_matrix

def generate_group_list_average(problem_count, m):
    base = m // problem_count
    remainder = m % problem_count
    return [base + (1 if i < remainder else 0) for i in range(problem_count)]

def gen_data(problem_count, m, n, k, quant_type_str):
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)
    group_list = generate_group_list_average(problem_count, m)

    if quant_type_str == "float8_e4m3fn":
        gen_func = gen_data_fp8_e4m3
    elif quant_type_str == "float8_e5m2":
        gen_func = gen_data_fp8_e5m2
    else:
        raise ValueError(f"不支持的量化类型: {quant_type_str}")

    a_fp8, a_scale, a_fp32 = gen_func(m, k, 1)

    b_fp8_list = []
    b_fp8_trans_list = []
    b_scale_list = []
    b_scale_trans_list = []
    b_fp32_list = []
    for _ in range(problem_count):
        b_fp8, b_scale, b_fp32 = gen_func(k, n, 0)
        b_scale_reshaped = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

        b_scale_nt = b_scale_reshaped.permute(0, 2, 1)
        b_fp8_trans = b_fp8.t().contiguous()
        b_scale_t = b_scale_reshaped.permute(2, 0, 1)

        b_fp8_list.append(b_fp8)
        b_fp8_trans_list.append(b_fp8_trans)
        b_scale_list.append(b_scale_nt)
        b_scale_trans_list.append(b_scale_t)
        b_fp32_list.append(b_fp32)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)

    b_fp8_all = torch.stack(b_fp8_list, dim=0)
    b_fp8_trans_all = torch.stack(b_fp8_trans_list, dim=0)
    b_scale_all = torch.stack(b_scale_list, dim=0)
    b_scale_trans_all = torch.stack(b_scale_trans_list, dim=0)
    b_fp32_all = torch.stack(b_fp32_list, dim=0)

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile(os.path.join(input_dir, "a_8.bin"))

    b_np = torch.tensor(b_fp8_all.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np.tofile(os.path.join(input_dir, "b_8.bin"))

    b_trans_np = torch.tensor(b_fp8_trans_all.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_trans_np.tofile(os.path.join(input_dir, "b_8_trans.bin"))

    a_scale_np = torch.tensor(a_scale.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))

    b_scale_np = torch.tensor(b_scale_all.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    b_scale_trans_np = torch.tensor(b_scale_trans_all.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_scale_trans_np.tofile(os.path.join(input_dir, "b_scale_trans.bin"))

    a_fp32_np = a_fp32.flatten().numpy()
    a_fp32_np.tofile(os.path.join(input_dir, "a_fp32.bin"))

    b_fp32_np = b_fp32_all.flatten().numpy()
    b_fp32_np.tofile(os.path.join(input_dir, "b_fp32.bin"))

    group_list_np = np.array(group_list, dtype=np.int64)
    group_list_np.tofile(os.path.join(input_dir, "group_list.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_count', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()

    if args.n % 128 != 0:
        raise ValueError(f"N ({args.n}) 必须128对齐")

    quant_type = 'float8_e4m3fn'
    gen_data(args.problem_count, args.m, args.n, args.k, quant_type)
