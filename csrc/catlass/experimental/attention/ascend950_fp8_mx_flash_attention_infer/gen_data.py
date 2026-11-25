#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import random
import torch
import math
from ml_dtypes import bfloat16
from dataclasses import dataclass
from typing import Tuple
np.random.seed(1)


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

    # FP8_E8M0FNU 缩放格式定义
    SCALE_FORMAT = {
        'name': 'FP8_E8M0FNU',
        'exp_bits': 8,
        'mantissa_bits': 0,
        'bias': 128,  # 偏置为128，0表示指数-128
        'max_exp': 127,
        'min_exp': -128,
        'max_value': 2**127,   # 约 1.7e38
        'min_value': 2**-128,  # 约 2.9e-39
        'signed': False,       # 无符号，仅正数
        'allow_zero': True     # 允许零值（指数-128表示零）
    }

    def __init__(self,
                 data_format: str = 'E4M3',
                 axis: int = 1,
                 block_size: int = 32,
                 epsilon: float = 1e-12):
        """
        初始化 MXFP8 矩阵量化器

        参数:
            data_format: 数据格式 'E4M3' 或 'E5M2'
            axis: 量化轴 (0: 行方向, 1: 列方向)
            block_size: 分块大小 (在量化轴上的元素数)
            epsilon: 防止除零的小值
        """
        if data_format not in self.DATA_FORMATS:
            raise ValueError(f"不支持的数据格式: {data_format}，支持: {list(self.DATA_FORMATS.keys())}")

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
        if self.data_format == 'E4M3':
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
                value = sign * self.config['max_value']
            else:
                exp = (val >> 3) & 0x0F
                mantissa = val & 0x07

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 8.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

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
                value = sign * self.config['max_value']
            else:
                exp = (val >> 2) & 0x1F
                mantissa = val & 0x03

                if exp == 0:
                    # 次正规数
                    value = (mantissa / 4.0) * (2.0 ** (1 - self.config['bias']))
                else:
                    # 正规数
                    value = (1.0 + mantissa / 4.0) * (2.0 ** (exp - self.config['bias']))

                value = sign * value

            # 钳位到有效范围
            if value > self.config['max_value']:
                value = self.config['max_value']
            elif value < self.config['min_value']:
                value = self.config['min_value']

            values.append(value)

        self.fp8_lut = torch.tensor(values, dtype=torch.float32)
        self.fp8_min = self.config['min_value']
        self.fp8_max = self.config['max_value']

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
        exp = int(math.floor(log2_scale)) - self.config['emax']

        # 钳位到 E8M0FNU 范围 [-128, 127]
        exp = max(self.SCALE_FORMAT['min_exp'],
                    min(exp, self.SCALE_FORMAT['max_exp']))

        # 计算缩放因子
        scale = 2.0 ** exp

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

        # 使用查找表量化
        quantized_flat = torch.zeros_like(data_clamped)

        # 向量化查找（比循环快）
        # 为每个值找到最接近的 FP8 值
        for i in range(len(data_clamped)):
            val = data_clamped[i].item()

            # 计算到所有 FP8 值的距离
            distances = torch.abs(self.fp8_lut - val)
            min_idx = torch.argmin(distances).item()

            quantized_flat[i] = self.fp8_lut[min_idx]

        return quantized_flat.view(original_shape)

    def _process_block(self, block_data: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
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

    def quantize_matrix(self,
                       matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _quantize_by_rows(self,
                         matrix: torch.Tensor,
                         M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _quantize_by_cols(self,
                         matrix: torch.Tensor,
                         M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def dequantize_matrix(self,
                         quantized_matrix: torch.Tensor,
                         scale_matrix: torch.Tensor) -> torch.Tensor:
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

    def _dequantize_by_rows(self,
                           quantized_matrix: torch.Tensor,
                           scale_matrix: torch.Tensor,
                           M: int, N: int) -> torch.Tensor:
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
                    dequantized_matrix[start_row:end_row, col] = (
                        quantized_matrix[start_row:end_row, col]
                    )

        return dequantized_matrix

    def _dequantize_by_cols(self,
                           quantized_matrix: torch.Tensor,
                           scale_matrix: torch.Tensor,
                           M: int, N: int) -> torch.Tensor:
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
                    dequantized_matrix[row, start_col:end_col] = (
                        quantized_matrix[row, start_col:end_col]
                    )

        return dequantized_matrix

def gen_data_fp8_e4m3(matrix_input, row, col, axis):
    matrix = torch.from_numpy(matrix_input).reshape(row,col)

    quantizer = MXFP8MatrixQuantizer(
        data_format='E4M3',
        axis=axis,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(
        quantized_matrix,
        scale_matrix
    )

    quantized_matrix = quantized_matrix.to(torch.float8_e4m3fn)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix

def gen_data_fp8_e5m2(matrix_input, row, col, axis):
    matrix = torch.from_numpy(matrix_input).reshape(row,col)

    quantizer = MXFP8MatrixQuantizer(
        data_format='E5M2',
        axis=axis,
        block_size=32
    )

    # 量化
    quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

    # 反量化
    dequantized_matrix = quantizer.dequantize_matrix(
        quantized_matrix,
        scale_matrix
    )

    quantized_matrix = quantized_matrix.to(torch.float8_e5m2)
    scale_matrix = scale_matrix.to(torch.float8_e8m0fnu)

    return quantized_matrix, scale_matrix, dequantized_matrix


def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for i in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(1, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    return q_seqlen_list, kv_seqlen_list

class TestFlashAttentionInfer():

    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any
        shape_param: any
        scale_q: any = None
        scale_k: any = None
        scale_v: any = None
        scale_p: any = None

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int
        use_fp8: bool = False
        use_p_scale: bool = False

    @classmethod
    def check_attr(cls, batch: int, q_seqlen: int, kv_seqlen: int, num_blocks: int, block_size: int):
        if q_seqlen > kv_seqlen:
            logging("[ERROR] q_seqlen cannot exceed kv_seqlen.")
            sys.exit()

    @classmethod
    def group_matmul(cls, head, kv_head, left, right):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :],
                                    right[i:(i + 1), :, :]).astype(np.float32)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention_mxfp8(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,   # (q_seqlen, k_seqlen)
            scale_p # (1)
    ):
        # Q * K.T
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        value = np.transpose(value, (1, 0, 2))

        out = np.zeros_like(query)
        post_mask_factor = -3e38
        for n_i in range(query.shape[0]):
            group_num = query.shape[0] // key.shape[0]
            kv_head_i = n_i // group_num
            num_s1_block = (query.shape[1] + 127) // 128
            for s1_i in range(num_s1_block):
                s1_start = s1_i * 128
                s1_end = (s1_i + 1) * 128 if  s1_i < num_s1_block - 1 else query.shape[1]
                s1_len = s1_end - s1_start
                max_i = np.ones((s1_len, 1)) * (-3e38)
                sum_i = np.zeros((s1_len, 1))
                o_i = np.zeros((s1_len, 1))
                num_s2_block = (key.shape[2] + 127) // 128
                for s2_i in range(num_s2_block):
                    s2_start = s2_i * 128
                    s2_end = (s2_i + 1) * 128 if  s2_i < num_s2_block - 1 else key.shape[2]
                    sblock = np.matmul(query[n_i, s1_start : s1_end, :],
                                    key[kv_head_i, :, s2_start : s2_end]).astype(np.float32)
                    sblock *= scale
                    if mask is not None:
                        sblock = sblock + (
                            mask[s1_start : s1_end, s2_start : s2_end]
                            ).astype(np.float32) * post_mask_factor
                    row_max = np.max(sblock, axis=-1, keepdims=True)
                    max_i_new = np.maximum(row_max, max_i)
                    pblock = np.exp(sblock - max_i_new)
                    if scale_p is not None:
                        pblock *= scale_p # add P dequant
                    sum_i = sum_i * np.exp(max_i - max_i_new) + np.sum(pblock, axis=-1, keepdims=True)
                    pblock_fp8 = torch.from_numpy(pblock)
                    pblock_fp8 = pblock_fp8.to(torch.float8_e4m3fn).to(torch.float)
                    pblock = pblock_fp8.numpy()

                    o_temp = np.matmul(pblock,
                                    value[kv_head_i, s2_start : s2_end, :]).astype(np.float32)
                    o_i = o_i * np.exp(max_i - max_i_new) + o_temp
                    
                    max_i = max_i_new
                o_i = o_i / sum_i
                out[n_i, s1_start : s1_end, :] = o_i

        out = np.transpose(out, (1, 0, 2))
        out_high = out.copy()
        out = out.astype(query.dtype)

        return out, out_high

    def ref_single_query_cached_kv_attention(self, attention_inputs: AttentionInputs, output, true_out) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size

        batch = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        kv_seqlen_now = 0
        scale_p = attention_inputs.scale_p
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            q = attention_inputs.query[cu_seqlen:(cu_seqlen + q_seqlen), :, :]
            keys = None
            values = None
            if attention_inputs.shape_param.kv_dtype == 1:
                keys = []
                values = []
                block_table = attention_inputs.block_tables[i]
                for j in range(k_seqlen): # j 每个k token拼接
                    block_number = int(block_table[j // block_size])
                    block_offset = j % block_size

                    k = attention_inputs.key_cache[block_number, block_offset, :, :]
                    k = k.reshape(kv_heads, head_size_qk)
                    keys.append(k)

                    v = attention_inputs.value_cache[block_number, block_offset, :, :]
                    v = v.reshape(kv_heads, head_size_vo)
                    values.append(v)
                keys = np.stack(keys, axis=0)
                values = np.stack(values, axis=0)
            elif attention_inputs.shape_param.kv_dtype == 0:
                keys = attention_inputs.key_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
                values = attention_inputs.value_cache[kv_seqlen_now: kv_seqlen_now + k_seqlen, :, :]
            scale = 1.0 / (head_size_qk ** 0.5)
            if attention_inputs.mask_type > 0:
                mask = attention_inputs.global_mask[cu_seqlen:(cu_seqlen + q_seqlen), :]
            else:
                mask = None
            out, out_high = self.ref_masked_attention_mxfp8(q, keys, values, scale, mask, scale_p)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += q_seqlen
            kv_seqlen_now += k_seqlen

    def calc_data(self, gen_data_params: GenDataParams):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        num_kv_tokens = np.array(gen_data_params.k_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        
        # 生成 FP32 数据
        query_fp32 = np.random.uniform(q_min_range, q_max_range,
            size=(num_tokens, gen_data_params.num_heads, head_size_qk)).astype(np.float32)
        key_cache_fp32 = None
        value_cache_fp32 = None
        
        block_tables = []   # (num_tokens, max_num_blocks_per_seq)
        layout = 'TND'
        if gen_data_params.kv_dtype == 1:
            key_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_qk)).astype(np.float32)

            value_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size,
                gen_data_params.kv_heads, head_size_vo)).astype(np.float32)
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(batch_size):
                block_table = [
                    max_num_blocks_per_seq * i + j
                    for j in range(max_num_blocks_per_seq)
                ]
                block_tables.append(block_table)
        elif gen_data_params.kv_dtype == 0:
            if layout == 'TND':
                key_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_qk)).astype(np.float32)
                value_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                    size=(num_kv_tokens, gen_data_params.kv_heads, head_size_vo)).astype(np.float32)
            elif layout == 'BSND':
                key_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_qk)).astype(np.float32)
                value_cache_fp32 = np.random.uniform(kv_min_range, kv_max_range,
                    size=(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_vo)).astype(np.float32)
        
        # 初始化量化后的数据
        scale_q = None
        scale_k = None
        scale_v = None
        
        if gen_data_params.use_fp8:
            # 量化 Q, M = num_tokens * gen_data_params.num_heads, K = head_size_qk, axis = 1
            q_fp8, scale_q, q_fp32_dequant = gen_data_fp8_e4m3(query_fp32, num_tokens * gen_data_params.num_heads, head_size_qk, 1)
            scale_q = scale_q.reshape(num_tokens, gen_data_params.num_heads, head_size_qk //64, 2)

            if gen_data_params.kv_dtype == 0 and layout == 'TND':
                # 量化 K, N = num_kv_tokens * gen_data_params.kv_heads, K = head_size_qk, axis = 1
                k_fp8, scale_k, k_fp32_dequant = gen_data_fp8_e4m3(key_cache_fp32, num_kv_tokens * gen_data_params.kv_heads, head_size_qk, 1)
                
                # 量化 V, K = num_kv_tokens, N = gen_data_params.kv_heads * head_size_vo, axis = 0
                v_fp8, scale_v, v_fp32_dequant = gen_data_fp8_e4m3(value_cache_fp32, num_kv_tokens, gen_data_params.kv_heads * head_size_vo, 0)

                scale_k = scale_k.reshape(num_kv_tokens, gen_data_params.kv_heads, head_size_qk //64, 2)
                scale_v = scale_v.reshape(num_kv_tokens //64, 2, gen_data_params.kv_heads, head_size_vo).permute(0, 2, 3, 1)
            
            elif gen_data_params.kv_dtype == 0 and layout == 'BSND':    
                # 量化 K, N = batch_size * max_k_seqlen * gen_data_params.kv_heads, K = head_size_qk, axis = 1
                k_fp8, scale_k, k_fp32_dequant = gen_data_fp8_e4m3(key_cache_fp32, batch_size * max_k_seqlen * gen_data_params.kv_heads, head_size_qk, 1)
                
                # 量化 V, K = batch_size * max_k_seqlen, N = gen_data_params.kv_heads * head_size_vo, axis = 0
                v_fp8, scale_v, v_fp32_dequant = gen_data_fp8_e4m3(value_cache_fp32, batch_size * max_k_seqlen, gen_data_params.kv_heads * head_size_vo, 0)

                scale_k = scale_k.reshape(batch_size, max_k_seqlen, gen_data_params.kv_heads, head_size_qk //64, 2)
                scale_v = scale_v.reshape(batch_size, max_k_seqlen //64, 2, gen_data_params.kv_heads, head_size_vo).permute(0, 1, 3, 4, 2)
            
            elif gen_data_params.kv_dtype == 1:
                # 量化 K, N = num_blocks * block_size, K = kv_heads * head_size_qk, axis = 1
                k_fp8, scale_k, k_fp32_dequant = gen_data_fp8_e4m3(key_cache_fp32, gen_data_params.num_blocks * gen_data_params.block_size,
                    gen_data_params.kv_heads * head_size_qk, 1)
                
                # 量化 V, K = num_blocks * block_size, N = kv_heads * head_size_vo, axis = 0
                v_fp8, scale_v, v_fp32_dequant = gen_data_fp8_e4m3(value_cache_fp32, gen_data_params.num_blocks * gen_data_params.block_size,
                    gen_data_params.kv_heads * head_size_vo, 0)

                scale_k = scale_k.reshape(gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_qk //64, 2)
                scale_v = scale_v.reshape(gen_data_params.num_blocks, gen_data_params.block_size //64, 2, gen_data_params.kv_heads, head_size_vo).permute(0, 1, 3, 4, 2)
                
            query = q_fp32_dequant.reshape(query_fp32.shape).numpy().astype(gen_data_params.dtype)
            key_cache = k_fp32_dequant.reshape(key_cache_fp32.shape).numpy().astype(gen_data_params.dtype)
            value_cache = v_fp32_dequant.reshape(value_cache_fp32.shape).numpy().astype(gen_data_params.dtype)
        
        if gen_data_params.mask_type > 0:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(gen_data_params.dtype)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = gen_data_params.q_seqlen_list[i]
                kseqlen = gen_data_params.k_seqlen_list[i]
                max_seq_len = max(qseqlen, kseqlen)
                tri = np.ones((max_seq_len, max_seq_len))
                tri = np.triu(tri, 1).astype(gen_data_params.dtype)
                if gen_data_params.mask_type == 1:
                    mask[pre_qseqlen : (pre_qseqlen + qseqlen), 0 : kseqlen] = tri[0 : qseqlen, 0 : kseqlen] #left up
                else:
                    mask[pre_qseqlen : (pre_qseqlen + qseqlen), max_seq_len - kseqlen: max_seq_len] = \
                        tri[max_seq_len - qseqlen : max_seq_len, max_seq_len - kseqlen : max_seq_len] #right down
                pre_qseqlen += qseqlen
            mask = mask.astype(gen_data_params.dtype)
        elif gen_data_params.mask_type == 0:
            mask = None

        if gen_data_params.use_p_scale:
            scale_p = np.random.uniform(0, 1, size=(1)).astype(np.float32)
        else:
            scale_p = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = np.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = np.zeros(shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(query, key_cache, value_cache, block_tables,
            gen_data_params.q_seqlen_list, gen_data_params.k_seqlen_list, mask, gen_data_params.mask_type, gen_data_params,
            scale_q, scale_k, scale_v, scale_p)
        
        self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
        )

        num_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
        num_kv_tokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_ntokens.bin"))
        
        # 保存 FP8 数据，参考示例文件的方式
        if gen_data_params.use_fp8 and scale_q is not None:
            # 保存为二进制文件（使用 untyped_storage() 方法）
            q_np = torch.tensor(q_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
            k_np = torch.tensor(k_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
            v_np = torch.tensor(v_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()

            scale_q_np = torch.tensor(scale_q.flatten().untyped_storage(), dtype=torch.int8).numpy()
            scale_k_np = torch.tensor(scale_k.flatten().untyped_storage(), dtype=torch.int8).numpy()
            scale_v_np = torch.tensor(scale_v.flatten().untyped_storage(), dtype=torch.int8).numpy()
            
            q_np.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
            k_np.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
            v_np.tofile(os.path.join(WORKSPACE, "data", "v.bin"))
            
            scale_q_np.tofile(os.path.join(WORKSPACE, "data", "scale_q.bin"))
            scale_k_np.tofile(os.path.join(WORKSPACE, "data", "scale_k.bin"))
            scale_v_np.tofile(os.path.join(WORKSPACE, "data", "scale_v.bin"))
            
            if gen_data_params.use_p_scale:
                scale_p.tofile(os.path.join(WORKSPACE, "data", "scale_p.bin"))
        else:
            query.tofile(os.path.join(WORKSPACE, "data", "q.bin"))
            key_cache.tofile(os.path.join(WORKSPACE, "data", "k.bin"))
            value_cache.tofile(os.path.join(WORKSPACE, "data", "v.bin"))
        
        np.array(block_tables).astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "block_table.bin"))
        np.array(gen_data_params.q_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
        np.array(gen_data_params.k_seqlen_list).astype(np.int64).tofile(
            os.path.join(WORKSPACE, "data", "kv_seqlen.bin"))
        if mask is not None:
            mask_out = mask.astype(np.uint8)
            mask_out.tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
        ref_output.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden.bin"))


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_head = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    block_size = 128
    is_varied_len = int(sys.argv[7])
    mask_type = int(sys.argv[8])
    kv_dtype = int(sys.argv[9])
    str_dtype = str(sys.argv[10])
    use_p_scale = int(sys.argv[11])
    if str_dtype == "half":
        dtype = np.float16
    elif str_dtype == "bf16":
        dtype = bfloat16
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()
    
    if use_p_scale != 0 and use_p_scale != 1:
        logging("[ERROR] p_scale_type must be 0 or 1")
        sys.exit()

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, batch)
    
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = batch * ((max_kv_seqlen + block_size - 1) // block_size)
    
    testObj = TestFlashAttentionInfer()
    gen_data_params = testObj.GenDataParams(q_seqlen_list, kv_seqlen_list, num_head,
                                            kv_heads, embedding_size,
                                            num_blocks, block_size, mask_type, dtype, kv_dtype,
                                            use_fp8=True, use_p_scale=use_p_scale==1)
    testObj.calc_data(gen_data_params)
