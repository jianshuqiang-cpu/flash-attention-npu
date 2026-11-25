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
"""
Generate data for example 63.

The standalone generator keeps the same quantization and golden path used during
bring-up: fp16 inputs, rint FP4 quantization, MX-only dequantization, and bf16
rounded C golden.
"""

import argparse
import math
import os
import sys
from typing import Dict, List

import numpy as np

try:
    from ml_dtypes import bfloat16
except ModuleNotFoundError:
    bfloat16 = None


np.random.seed(42)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE, "data")

FP4_E2M1_MAX = 6.0
FP4_E2M1_EMAX = 2
LEVEL0_BLOCK_SIZE = 512
LEVEL1_BLOCK_SIZE = 32
MX_SCALE_GROUP_NUM = 32


def round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def cast_to_bfloat16_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if bfloat16 is not None:
        return values.astype(bfloat16).astype(np.float32)

    bits = values.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + (np.uint32(0x7FFF) + lsb)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


class DualLevelMXFP4Quantizer:
    FP4_E2M1_LUT = None
    FP4_E2M1_LUT_UNSIGNED = None

    @classmethod
    def _build_fp4_e2m1_lut(cls):
        if cls.FP4_E2M1_LUT is not None:
            return

        cls.FP4_E2M1_LUT_UNSIGNED = []
        for i in range(8):
            exp = (i >> 1) & 0x03
            mantissa = i & 0x01
            bias = 1
            if exp == 0:
                value = 0.0 if mantissa == 0 else (mantissa / 2.0) * (2.0 ** (1 - bias))
            else:
                value = (1.0 + mantissa / 2.0) * (2.0 ** (exp - bias))
            cls.FP4_E2M1_LUT_UNSIGNED.append(value)
        cls.FP4_E2M1_LUT_UNSIGNED = np.array(cls.FP4_E2M1_LUT_UNSIGNED, dtype=np.float32)

        cls.FP4_E2M1_LUT = np.zeros(16, dtype=np.float32)
        for i in range(8):
            cls.FP4_E2M1_LUT[i] = cls.FP4_E2M1_LUT_UNSIGNED[i]
            cls.FP4_E2M1_LUT[i + 8] = -cls.FP4_E2M1_LUT_UNSIGNED[i]

    def __init__(self):
        self._build_fp4_e2m1_lut()

    @staticmethod
    def cast_to_kernel_xtmp(values: np.ndarray) -> np.ndarray:
        return values.astype(np.float16).astype(np.float32)

    def quantize_fp4_e2m1_batch(self, values: np.ndarray, round_mode: str = "current") -> np.ndarray:
        if round_mode == "rint":
            return self.quantize_fp4_e2m1_rint_batch(values)
        if round_mode != "current":
            raise ValueError(f"Unsupported FP4 round mode: {round_mode}")

        values = np.asarray(values)
        shape = values.shape
        flat = np.clip(values.reshape(-1), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        abs_values = np.abs(flat)
        signs = (flat < 0).astype(np.int8)
        distances = np.abs(abs_values[:, np.newaxis] - self.FP4_E2M1_LUT_UNSIGNED[np.newaxis, :])
        min_idx = np.argmin(distances, axis=1)
        fp4_codes = (min_idx + signs * 8).astype(np.uint8)
        return fp4_codes.reshape(shape)

    def quantize_fp4_e2m1_rint_batch(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        shape = values.shape
        values_f32 = values.astype(np.float32).reshape(-1)
        sign = np.signbit(values_f32).astype(np.uint8)
        abs_values = np.minimum(np.abs(values_f32).astype(np.float64), FP4_E2M1_MAX)

        lut = self.FP4_E2M1_LUT_UNSIGNED.astype(np.float64)
        hi = np.searchsorted(lut, abs_values, side="right")
        hi = np.clip(hi, 1, len(lut) - 1)
        lo = hi - 1

        lo_val = lut[lo]
        hi_val = lut[hi]
        mid = (lo_val + hi_val) / 2.0
        lo_is_even = (lo & 1) == 0
        pick_lo = (abs_values < mid) | ((abs_values == mid) & lo_is_even)
        chosen_idx = np.where(pick_lo, lo, hi).astype(np.uint8)
        codes = (chosen_idx | (sign << 3)).astype(np.uint8)
        return codes.reshape(shape)

    @staticmethod
    def _compute_fp8_e8m0_scale_vec(max_abs: np.ndarray) -> tuple:
        max_abs = np.asarray(max_abs)
        zero_mask = max_abs < 1e-12
        safe_abs = np.where(zero_mask, 1.0, max_abs)
        log2_val = np.log2(safe_abs)
        exp = np.floor(log2_val).astype(np.int64) - FP4_E2M1_EMAX
        exp = np.clip(exp, -128, 127)
        scale_float = np.where(
            zero_mask,
            np.float64(1.0),
            np.ldexp(np.float64(1.0), exp.astype(np.int32)),
        )
        scale_u8 = np.where(zero_mask, np.uint8(127), (exp + 127).astype(np.uint8))
        return scale_float, scale_u8.astype(np.uint8)

    @staticmethod
    def _pack_fp4_codes(codes: np.ndarray) -> np.ndarray:
        c = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
        return (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)

    @staticmethod
    def _unpack_fp4_codes(packed: np.ndarray, k: int) -> np.ndarray:
        low = (packed & 0x0F).astype(np.uint8)
        high = ((packed >> 4) & 0x0F).astype(np.uint8)
        unpacked = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
        return unpacked[..., :k]

    def _shared_quantize_rows(self, matrix: np.ndarray) -> tuple:
        matrix = np.asarray(matrix, dtype=np.float32)
        rows, k = matrix.shape
        if k % 2 != 0:
            raise ValueError(f"K must be even for FP4 packing: k={k}")

        num_l0 = (k + LEVEL0_BLOCK_SIZE - 1) // LEVEL0_BLOCK_SIZE
        num_l1 = (k + LEVEL1_BLOCK_SIZE - 1) // LEVEL1_BLOCK_SIZE
        k_padded = num_l0 * LEVEL0_BLOCK_SIZE
        num_l1_padded = k_padded // LEVEL1_BLOCK_SIZE

        if k_padded != k:
            work = np.zeros((rows, k_padded), dtype=np.float32)
            work[:, :k] = matrix
        else:
            work = matrix

        l0 = work.reshape(rows, num_l0, LEVEL0_BLOCK_SIZE)
        max_abs0 = np.max(np.abs(l0), axis=-1)
        scale0 = np.where(
            max_abs0 < 1e-12,
            np.float32(1.0),
            (max_abs0 / FP4_E2M1_MAX).astype(np.float32),
        ).astype(np.float32)

        scale0_expanded = np.repeat(scale0, LEVEL0_BLOCK_SIZE, axis=1)
        temp = self.cast_to_kernel_xtmp(work / scale0_expanded)

        l1 = temp.reshape(rows, num_l1_padded, LEVEL1_BLOCK_SIZE)
        max_abs1 = np.max(np.abs(l1), axis=-1)
        scale1_f_padded, scale1_u8_padded = self._compute_fp8_e8m0_scale_vec(max_abs1)

        scaled_padded = l1.astype(np.float64) / scale1_f_padded[..., np.newaxis]
        scaled_padded = scaled_padded.reshape(rows, k_padded)

        return scale0, scale1_u8_padded[:, :num_l1], scaled_padded[:, :k]

    def _finalize_quant(self, scale0: np.ndarray, scale1_u8: np.ndarray,
                        scaled: np.ndarray, fp4_round_mode: str) -> dict:
        codes = self.quantize_fp4_e2m1_batch(scaled, fp4_round_mode)
        quant_data = self._pack_fp4_codes(codes)
        rows, k = scaled.shape
        return {
            "quant_data": quant_data,
            "scale0": scale0,
            "scale1": scale1_u8,
            "M": rows,
            "N": rows,
            "K": k,
        }

    def dual_level_quantize_pair_a(self, matrix: np.ndarray) -> tuple:
        scale0, scale1_u8, scaled = self._shared_quantize_rows(matrix)
        current = self._finalize_quant(scale0, scale1_u8, scaled, "current")
        rint = self._finalize_quant(scale0, scale1_u8, scaled, "rint")
        for result in (current, rint):
            result.pop("N", None)
        return current, rint

    def dual_level_quantize_pair_b(self, matrix: np.ndarray) -> tuple:
        k, n = matrix.shape
        scale0, scale1_u8, scaled = self._shared_quantize_rows(matrix.T)
        current = self._finalize_quant(scale0, scale1_u8, scaled, "current")
        rint = self._finalize_quant(scale0, scale1_u8, scaled, "rint")
        for result in (current, rint):
            result["K"] = k
            result["N"] = n
            result.pop("M", None)
        return current, rint

    def dequantize_a_mx_only(self, quant_data: np.ndarray, scale1: np.ndarray,
                             m: int, k: int) -> np.ndarray:
        unpacked = self._unpack_fp4_codes(quant_data, k)
        fp4_vals = self.FP4_E2M1_LUT[unpacked]
        exp = scale1.astype(np.int32) - 127
        sf = np.ldexp(np.float64(1.0), exp)
        sf_expanded = np.repeat(sf, LEVEL1_BLOCK_SIZE, axis=1)[:, :k]
        return (fp4_vals.astype(np.float64) * sf_expanded).astype(np.float32)

    def dequantize_b_mx_only(self, quant_data: np.ndarray, scale1: np.ndarray,
                             k: int, n: int) -> np.ndarray:
        unpacked = self._unpack_fp4_codes(quant_data, k)
        fp4_vals = self.FP4_E2M1_LUT[unpacked]
        exp = scale1.astype(np.int32) - 127
        sf = np.ldexp(np.float64(1.0), exp)
        sf_expanded = np.repeat(sf, LEVEL1_BLOCK_SIZE, axis=1)[:, :k]
        dequant_t = (fp4_vals.astype(np.float64) * sf_expanded).astype(np.float32)
        return dequant_t.T.copy()


def compute_mx_matmul_golden(quant_a, scale1_a, quant_b, scale1_b, m, n, k):
    quantizer = DualLevelMXFP4Quantizer()
    dequant_a = quantizer.dequantize_a_mx_only(quant_a, scale1_a, m, k)
    dequant_b = quantizer.dequantize_b_mx_only(quant_b, scale1_b, k, n)
    result = np.matmul(dequant_a, dequant_b)
    return cast_to_bfloat16_float32(result)


def save_binary_data(data_dir: str, name: str, data: np.ndarray, dtype=None):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, name)
    arr = data.astype(dtype) if dtype else data
    arr.tofile(filepath)


def validate_case(batch: int, m: int, n: int, k: int) -> str:
    if batch <= 0 or m <= 0 or n <= 0 or k <= 0:
        return f"batch/m/n/k must be positive integers: batch={batch}, m={m}, n={n}, k={k}"
    if k % 2 != 0:
        return f"K must be even for FP4 packing: k={k}"
    return ""


def gen_data(batch: int, m: int, n: int, k: int) -> List[Dict]:
    print(f"\n{'=' * 60}")
    print(f"  [Step 1] 生成测试数据: Batch={batch}, M={m}, N={n}, K={k}")
    print(f"{'=' * 60}")

    quantizer = DualLevelMXFP4Quantizer()
    input_dir = os.path.join(DATA_DIR, "input")
    golden_dir = os.path.join(DATA_DIR, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    input_a_list = []
    input_b_list = []
    golden_c_list = []
    scale_a1_list = []
    scale_a2_list = []
    scale_b1_list = []
    scale_b2_list = []
    quant_a_current_list = []
    quant_b_current_list = []
    quant_a_rint_list = []
    quant_b_rint_list = []
    results = []

    for batch_idx in range(batch):
        a_fp16 = np.random.randn(m, k).astype(np.float16) * 2.0
        b_fp16 = np.random.randn(k, n).astype(np.float16) * 2.0

        a_fp32 = a_fp16.astype(np.float32)
        b_fp32 = b_fp16.astype(np.float32)

        result_a_current, result_a_rint = quantizer.dual_level_quantize_pair_a(a_fp32)
        result_b_current, result_b_rint = quantizer.dual_level_quantize_pair_b(b_fp32)
        result_a = result_a_rint
        result_b = result_b_rint

        quant_a = result_a["quant_data"]
        scale0_a = result_a["scale0"]
        scale1_a = result_a["scale1"]
        quant_b = result_b["quant_data"]
        scale0_b = result_b["scale0"]
        scale1_b = result_b["scale1"]

        golden_c = compute_mx_matmul_golden(quant_a, scale1_a, quant_b, scale1_b, m, n, k)
        golden_fp32 = np.matmul(a_fp32, b_fp32)

        input_a_list.append(a_fp16)
        input_b_list.append(b_fp16.T)
        golden_c_list.append(golden_c)
        scale_a1_list.append(scale0_a)
        scale_a2_list.append(scale1_a)
        scale_b1_list.append(scale0_b)
        scale_b2_list.append(scale1_b)
        quant_a_current_list.append(result_a_current["quant_data"])
        quant_b_current_list.append(result_b_current["quant_data"])
        quant_a_rint_list.append(result_a_rint["quant_data"])
        quant_b_rint_list.append(result_b_rint["quant_data"])

        results.append({
            "batch_idx": batch_idx,
            "a_fp32": a_fp32,
            "b_fp32": b_fp32,
            "quant_a": quant_a,
            "quant_b": quant_b,
            "quant_a_current": result_a_current["quant_data"],
            "quant_b_current": result_b_current["quant_data"],
            "quant_a_rint": result_a_rint["quant_data"],
            "quant_b_rint": result_b_rint["quant_data"],
            "scale1_a": scale1_a,
            "scale1_b": scale1_b,
            "golden_c": golden_c,
            "golden_fp32": golden_fp32,
        })
        print(f"  batch_{batch_idx}: data generated")

    input_a_batch = np.stack(input_a_list, axis=0)
    input_b_batch = np.stack(input_b_list, axis=0)
    golden_c_batch = np.stack(golden_c_list, axis=0)
    scale_a1_batch = np.stack(scale_a1_list, axis=0)
    scale_a2_batch = np.stack(scale_a2_list, axis=0)
    scale_b1_batch = np.stack(scale_b1_list, axis=0)
    scale_b2_batch = np.stack(scale_b2_list, axis=0)
    quant_a_current_batch = np.stack(quant_a_current_list, axis=0)
    quant_b_current_batch = np.stack(quant_b_current_list, axis=0)
    quant_a_rint_batch = np.stack(quant_a_rint_list, axis=0)
    quant_b_rint_batch = np.stack(quant_b_rint_list, axis=0)

    mx_scale_k = (k + MX_SCALE_GROUP_NUM - 1) // MX_SCALE_GROUP_NUM
    mx_scale_aligned_k = round_up(mx_scale_k, 2)
    if mx_scale_aligned_k != mx_scale_k:
        pad_width = ((0, 0), (0, 0), (0, mx_scale_aligned_k - mx_scale_k))
        scale_a2_batch = np.pad(scale_a2_batch, pad_width, mode="constant", constant_values=0)
        scale_b2_batch = np.pad(scale_b2_batch, pad_width, mode="constant", constant_values=0)

    save_binary_data(input_dir, "input_a.bin", input_a_batch, np.float16)
    save_binary_data(input_dir, "input_b.bin", input_b_batch, np.float16)
    save_binary_data(golden_dir, "expected_data.bin", golden_c_batch, np.float32)
    save_binary_data(golden_dir, "expected_scale_a1.bin", scale_a1_batch, np.float32)
    save_binary_data(golden_dir, "expected_scale_a2.bin", scale_a2_batch, np.uint8)
    save_binary_data(golden_dir, "expected_scale_b1.bin", scale_b1_batch, np.float32)
    save_binary_data(golden_dir, "expected_scale_b2.bin", scale_b2_batch, np.uint8)
    save_binary_data(golden_dir, "expected_quant_a_current.bin", quant_a_current_batch, np.uint8)
    save_binary_data(golden_dir, "expected_quant_b_current.bin", quant_b_current_batch, np.uint8)
    save_binary_data(golden_dir, "expected_quant_a_rint.bin", quant_a_rint_batch, np.uint8)
    save_binary_data(golden_dir, "expected_quant_b_rint.bin", quant_b_rint_batch, np.uint8)

    print(f"  Input data saved to {input_dir}")
    print(f"  Golden data saved to {golden_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate data for dual-level quant MX batch matmul."
    )
    parser.add_argument("batch", type=int, help="Batch count")
    parser.add_argument("m", type=int, help="M dimension")
    parser.add_argument("n", type=int, help="N dimension")
    parser.add_argument("k", type=int, help="K dimension")
    parser.add_argument("trans_a", type=int, nargs="?", default=0,
                        help="Accepted for compatibility; generated layout is fixed.")
    parser.add_argument("trans_b", type=int, nargs="?", default=1,
                        help="Accepted for compatibility; generated layout is fixed.")
    args = parser.parse_args()

    error_msg = validate_case(args.batch, args.m, args.n, args.k)
    if error_msg:
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return 2

    if args.trans_a != 0 or args.trans_b != 1:
        print("[WARN] trans_a/trans_b are accepted only for CLI compatibility; "
              "example 63 always stores A as [batch,M,K] and B physically as [batch,N,K].")

    try:
        generated_results = gen_data(args.batch, args.m, args.n, args.k)
    except Exception as exc:
        print(f"[ERROR] Failed to generate data: {exc}", file=sys.stderr)
        return 1

    if len(generated_results) != args.batch:
        print(
            f"[ERROR] Generated batch count mismatch: "
            f"expected {args.batch}, got {len(generated_results)}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
