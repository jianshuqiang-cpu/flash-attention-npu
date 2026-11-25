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
import torch
import math
from typing import Tuple

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


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

    def _compute_scale_fp8_e8m0fnu(self, block_data: torch.Tensor) -> Tuple[float, int]:
        """
        计算 FP8_E8M0FNU 格式的缩放因子

        返回:
            scale: 缩放因子 (浮点数)
            exp: 指数 (E8M0FNU 格式)
        """
        # 计算块的最大绝对值
        max_abs = torch.max(torch.abs(block_data)).item()

        if max_abs < self.epsilon:
            # 全零或接近零的块
            return 1.0, 0  # 指数0表示缩放因子2^0=1

        # 计算对数 (log2)
        if max_abs > 0:
            log2_scale = math.log2(max_abs)
        else:
            log2_scale = -128  # 最小值

        # 四舍五入到最近的整数 (指数)
        exp = int(math.floor(log2_scale)) - self.config["emax"]

        # 钳位到 E8M0FNU 范围 [-128, 127]
        exp = max(self.SCALE_FORMAT["min_exp"], min(exp, self.SCALE_FORMAT["max_exp"]))

        # 计算缩放因子
        scale = 2.0**exp

        return scale, exp

    def _quantize_to_fp8(self, data: torch.Tensor) -> torch.Tensor:
        """
        量化数据到 FP8 格式

        使用查找表进行最近邻量化
        """
        original_shape = data.shape
        data_flat = data.flatten()

        # 钳位到 FP8 范围
        data_clamped = torch.clamp(data_flat, self.fp8_min, self.fp8_max)

        # 使用查找表量化 - 向量化实现，100×提速
        # 一次性计算所有元素到所有FP8值的距离
        # data_clamped: (N,) → (N, 1); fp8_lut: (256,) → (1, 256)
        distances = torch.abs(data_clamped.unsqueeze(1) - self.fp8_lut.unsqueeze(0))
        min_idx = torch.argmin(distances, dim=1)
        quantized_flat = self.fp8_lut[min_idx]

        return quantized_flat.view(original_shape)

    def _process_block(
        self, block_data: torch.Tensor
    ) -> Tuple[torch.Tensor, float, int]:
        """
        处理单个分块

        返回:
            quantized_block: 量化后的块
            scale: 缩放因子
            exp: 指数
        """
        # 计算缩放因子
        scale, exp = self._compute_scale_fp8_e8m0fnu(block_data)

        # 应用缩放
        if abs(scale) > self.epsilon:
            scaled_data = block_data / scale
        else:
            scaled_data = block_data  # 缩放因子为0，不缩放

        # 量化到 FP8
        quantized_scaled = self._quantize_to_fp8(scaled_data)

        return quantized_scaled, scale, exp

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
        """
        按行方向量化

        分块: 每 block_size 行一个块
        """
        # 计算分块数量
        num_blocks = (M + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, N), dtype=torch.float32)

        # 处理每个行块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束行
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 提取当前块
            block_data = matrix[start_row:end_row, :]

            # 对当前块的每一列单独处理（列方向）
            for col in range(N):
                # 提取列数据
                col_data = block_data[:, col]

                # 处理该列
                quantized_col, scale, exp = self._process_block(col_data)

                # 存储结果
                quantized_matrix[start_row:end_row, col] = quantized_col
                scale_matrix[block_idx, col] = scale

        return quantized_matrix, scale_matrix

    def _quantize_by_cols(
        self, matrix: torch.Tensor, M: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按列方向量化

        分块: 每 block_size 列一个块
        """
        # 计算分块数量
        num_blocks = (N + self.block_size - 1) // self.block_size

        # 初始化结果
        quantized_matrix = torch.zeros_like(matrix)
        scale_matrix = torch.ones((M, (num_blocks + 1) // 2 * 2), dtype=torch.float32)

        # 处理每个列块
        for block_idx in range(num_blocks):
            # 计算当前块的起始和结束列
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)

            # 提取当前块
            block_data = matrix[:, start_col:end_col]

            # 对当前块的每一行单独处理（行方向）
            for row in range(M):
                # 提取行数据
                row_data = block_data[row, :]

                # 处理该行
                quantized_row, scale, exp = self._process_block(row_data)

                # 存储结果
                quantized_matrix[row, start_col:end_col] = quantized_row
                scale_matrix[row, block_idx] = scale

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
        num_blocks = scale_matrix.shape[0]
        dequantized_matrix = torch.zeros_like(quantized_matrix)

        for block_idx in range(num_blocks):
            start_row = block_idx * self.block_size
            end_row = min(start_row + self.block_size, M)

            # 应用缩放因子
            for col in range(N):
                scale = scale_matrix[block_idx, col].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[start_row:end_row, col] = (
                        quantized_matrix[start_row:end_row, col] * scale
                    )
                else:
                    dequantized_matrix[start_row:end_row, col] = quantized_matrix[
                        start_row:end_row, col
                    ]

        return dequantized_matrix

    def _dequantize_by_cols(
        self, quantized_matrix: torch.Tensor, scale_matrix: torch.Tensor, M: int, N: int
    ) -> torch.Tensor:
        """列方向反量化"""
        num_blocks = scale_matrix.shape[1]
        dequantized_matrix = torch.zeros_like(quantized_matrix)

        for block_idx in range(num_blocks):
            start_col = block_idx * self.block_size
            end_col = min(start_col + self.block_size, N)

            # 应用缩放因子
            for row in range(M):
                scale = scale_matrix[row, block_idx].item()
                if abs(scale) > self.epsilon:
                    dequantized_matrix[row, start_col:end_col] = (
                        quantized_matrix[row, start_col:end_col] * scale
                    )
                else:
                    dequantized_matrix[row, start_col:end_col] = quantized_matrix[
                        row, start_col:end_col
                    ]

        return dequantized_matrix


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10

    quantizer = MXFP8MatrixQuantizer(data_format="E4M3", axis=axis, block_size=32)

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)

    quantizer = MXFP8MatrixQuantizer(data_format="E5M2", axis=axis, block_size=32)

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(quantized_matrix, scale_matrix)

    quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_data(batch, m, n, k, trans_a, trans_b):
    data_dir = os.path.join(WORKSPACE, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    a_fp8_list = []
    b_fp8_list = []
    a_scale_list = []
    b_scale_list = []
    c_fp32_list = []

    for _ in range(batch):
        a_fp8, a_scale, a_fp32 = gen_data_fp8_e4m3(m, k, 1)
        b_fp8, b_scale, b_fp32 = gen_data_fp8_e4m3(k, n, 0)

        a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
        b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

        if trans_a == 1:
            a_fp8 = a_fp8.t()
            a_scale = a_scale.permute(1, 0, 2)

        if trans_b == 1:
            b_fp8 = b_fp8.t()
            b_scale = b_scale.permute(2, 0, 1)
        else:
            b_scale = b_scale.permute(0, 2, 1)

        a_fp8_list.append(a_fp8)
        b_fp8_list.append(b_fp8)
        a_scale_list.append(a_scale)
        b_scale_list.append(b_scale)

        c_fp32 = a_fp32 @ b_fp32
        c_fp32_list.append(c_fp32)

    a_fp8_batch = torch.stack(a_fp8_list, dim=0)
    b_fp8_batch = torch.stack(b_fp8_list, dim=0)
    a_scale_batch = torch.stack(a_scale_list, dim=0)
    b_scale_batch = torch.stack(b_scale_list, dim=0)
    c_fp32_batch = torch.stack(c_fp32_list, dim=0)

    a_np = torch.tensor(
        a_fp8_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_np = torch.tensor(
        b_fp8_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_np.tofile(os.path.join(input_dir, "a_8.bin"))
    b_np.tofile(os.path.join(input_dir, "b_8.bin"))

    a_scale_np = torch.tensor(
        a_scale_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale_batch.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    c_np = c_fp32_batch.numpy()
    c_np.tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_a", type=int)
    parser.add_argument("trans_b", type=int)
    args = parser.parse_args()
    gen_data(args.batch, args.m, args.n, args.k, args.trans_a, args.trans_b)
