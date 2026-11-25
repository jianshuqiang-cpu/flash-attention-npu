#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broadcast Matmul Perblock Quant 数据生成与比对脚本
功能：
- 生成测试数据 (A, B)
- 计算golden结果 (dst fp8, scale)
- 运行NPU算子
- 比对dst和scale精度
"""

import os
import argparse
import subprocess
from typing import Tuple

import torch
import numpy as np
from ml_dtypes import float8_e4m3fn


E4M3_MAX = 448.0
E4M3_MIN = -448.0


def build_e4m3_lut():
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
            value = sign * E4M3_MAX
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                value = (mantissa / 8.0) * (2.0 ** (1 - 7))
            else:
                value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7))
            value = sign * value

        if value > E4M3_MAX:
            value = E4M3_MAX
        elif value < E4M3_MIN:
            value = E4M3_MIN
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


E4M3_LUT = build_e4m3_lut()


def quantize_to_e4m3(data: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(data, E4M3_MIN, E4M3_MAX)
    dist = torch.abs(clamped.unsqueeze(-1) - E4M3_LUT)
    indices = torch.argmin(dist, dim=-1)
    return indices.to(torch.uint8)


def compute_scale_and_quantize(matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
    abs_max = torch.amax(torch.abs(matrix))
    scale = torch.clamp(abs_max, min=1e-12) / E4M3_MAX
    quantized = quantize_to_e4m3(matrix / scale)
    return quantized, scale.item()


def compute_broadcast_matmul_perblock_quant(
    batch_count: int,
    m: int,
    n: int,
    k: int,
    host_a_bf16: torch.Tensor,
    host_b_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    host_a_fp32 = host_a_bf16.float()
    host_b_fp32 = host_b_bf16.float()
    a_reshaped_bf16 = host_a_bf16.view(batch_count, m, k)
    c_bf16 = torch.matmul(a_reshaped_bf16, host_b_bf16)

    a_reshaped_fp32 = host_a_fp32.view(batch_count, m, k)
    c_fp32 = torch.matmul(a_reshaped_fp32, host_b_fp32)

    dst_fp8 = torch.zeros(batch_count, m, n, dtype=torch.uint8)
    scale = torch.zeros(batch_count, dtype=torch.float32)
    dst_fp8_upgrade = torch.zeros(batch_count, m, n, dtype=torch.uint8)
    scale_upgrade = torch.zeros(batch_count, dtype=torch.float32)

    for b in range(batch_count):
        block_slice = c_bf16[b]
        quantized, sc = compute_scale_and_quantize(block_slice.float())
        dst_fp8[b] = quantized
        scale[b] = sc

        block_slice_upgrade = c_fp32[b]
        quantized_upgrade, sc_upgrade = compute_scale_and_quantize(block_slice_upgrade)
        dst_fp8_upgrade[b] = quantized_upgrade
        scale_upgrade[b] = sc_upgrade

    return dst_fp8, scale, dst_fp8_upgrade, scale_upgrade


def gen_data(batch_count: int, m: int, n: int, k: int):
    os.makedirs("./data", exist_ok=True)

    a_bf16 = torch.randn(batch_count, m, k, dtype=torch.bfloat16) * 2.0
    b_bf16 = torch.randn(k, n, dtype=torch.bfloat16) * 2.0

    a_uint16 = a_bf16.view(torch.uint16)
    b_uint16 = b_bf16.view(torch.uint16)

    a_np = a_uint16.numpy()
    b_np = b_uint16.numpy()
    a_np.tofile("./data/a.bin")
    b_np.tofile("./data/b.bin")

    dst_fp8, scale, dst_fp8_upgrade, scale_upgrade = (
        compute_broadcast_matmul_perblock_quant(batch_count, m, n, k, a_bf16, b_bf16)
    )

    dst_np = dst_fp8.numpy()
    dst_np.tofile("./data/dst.bin")

    scale_np = scale.numpy()
    scale_np.tofile("./data/scale.bin")

    return dst_fp8, scale, dst_fp8_upgrade, scale_upgrade


def compute_rela_errors(result, golden, mask=None):
    relative_errors = np.abs((result - golden) / (golden + 1e-7))
    mse = np.sqrt(np.mean((result - golden) ** 2))
    if mask is not None:
        if np.any(mask):
            filtered_max = np.max(relative_errors[mask])
            filtered_mean = np.mean(relative_errors[mask])
        else:
            filtered_max, filtered_mean = 0.0, 0.0
    else:
        filtered_max = np.max(relative_errors)
        filtered_mean = np.mean(relative_errors)

    return filtered_max, filtered_mean, mse


def compare_mare(mare_npu, mare_upgrade, dtype):
    if dtype == "half":
        err = 2 ** (-11)
    elif dtype == "bfloat16":
        err = 2 ** (-11)
    elif dtype == "float8_e4m3fn":
        err = 2 ** (-6)
    else:
        err = 2 ** (-12)
    res = True if (mare_npu / max(mare_upgrade, err)) < 10 else False
    return res


def compare_mere(mere_npu, mere_upgrade, dtype):
    if dtype == "half":
        err = 2 ** (-11)
    elif dtype == "bfloat16":
        err = 2 ** (-11)
    elif dtype == "float8_e4m3fn":
        err = 2 ** (-6)
    else:
        err = 2 ** (-12)
    res = True if (mere_npu / max(mere_upgrade, err)) < 2 else False
    return res


def compare_rmse(rmse_npu, rmse_upgrade, dtype):
    if dtype == "half":
        err = 2 ** (-11)
    elif dtype == "bfloat16":
        err = 2 ** (-6)
    elif dtype == "float8_e4m3fn":
        err = 2 ** (-2)
    else:
        err = 2 ** (-12)
    res = True if (rmse_npu / max(rmse_upgrade, err)) < 2 else False
    return res


def compare_small_data(
    result_dst_flat, golden_dst_flat, golden_dst_upgrade_flat, dst_mask
):
    ERR_THRESHOLD = 2 ** (-6)
    small_dst_mask = ~dst_mask

    golden_diff = np.abs(golden_dst_flat - golden_dst_upgrade_flat)
    error_count_golden_small = np.sum((small_dst_mask & (golden_diff > ERR_THRESHOLD)))

    result_diff = np.abs(result_dst_flat - golden_dst_upgrade_flat)
    error_count_result_small = np.sum((small_dst_mask & (result_diff > ERR_THRESHOLD)))

    ratio_small = error_count_result_small / max(error_count_golden_small, 1)
    return ratio_small < 2


def compare_results(result: np.ndarray, golden: np.ndarray, name: str):
    if result.shape != golden.shape:
        print(f"{name}: shape mismatch result={result.shape} golden={golden.shape}")
        return 1

    is_close = np.isclose(result, golden, rtol=1e-3, atol=1e-3)
    error_count = np.sum(~is_close)
    total_count = result.size
    accuracy = (total_count - error_count) / total_count * 100

    print(f"{name}: error_count={error_count}/{total_count} (accuracy={accuracy:.4f}%)")
    return error_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_count", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("device_id", type=int)
    args = parser.parse_args()

    batch_count = args.batch_count
    m = args.m
    n = args.n
    k = args.k
    device_id = args.device_id

    print("------ 生成测试数据 ------")
    print(f"batch_count={batch_count}, m={m}, n={n}, k={k}")

    golden_dst, golden_scale, golden_dst_upgrade, golden_scale_upgrade = gen_data(
        batch_count, m, n, k
    )

    print("------ 运行NPU算子 ------")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    catlass_home_dir = os.path.dirname(os.path.dirname(current_dir))
    op_path = os.path.join(
        catlass_home_dir,
        "output",
        "bin",
        "62_ascend950_broadcast_matmul_perblock_quant",
    )

    result = subprocess.run(
        [op_path, str(batch_count), str(m), str(n), str(k), str(device_id)],
        capture_output=True,
        text=True,
    )
    print(f"npu op run log = {result.stdout}")
    if result.stderr:
        print(f"npu op run err = {result.stderr}")

    print("------ 比对结果 ------")
    result_dst = np.fromfile("./data/result_dst.bin", dtype=np.uint8).reshape(
        batch_count, m, n
    )
    result_scale = np.fromfile("./data/result_scale.bin", dtype=np.float32)

    dtype = "bfloat16"

    print("------ 计算相对误差 -----")
    result_dst_flat = result_dst.flatten().view(float8_e4m3fn).astype(np.float32)
    golden_dst_flat = (
        golden_dst.numpy().flatten().view(float8_e4m3fn).astype(np.float32)
    )
    golden_dst_upgrade_flat = (
        golden_dst_upgrade.numpy().flatten().view(float8_e4m3fn).astype(np.float32)
    )
    SMALL_THRESHOLD = 2 ** (-6)
    dst_mask = (
        np.minimum(np.abs(result_dst_flat), np.abs(golden_dst_upgrade_flat))
        > SMALL_THRESHOLD
    )
    mare_dst, mere_dst, rmse_dst = compute_rela_errors(
        result_dst_flat, golden_dst_upgrade_flat, dst_mask
    )
    dst_mask_golden = (
        np.minimum(np.abs(golden_dst_flat), np.abs(golden_dst_upgrade_flat))
        > SMALL_THRESHOLD
    )
    mare_golden, mere_golden, rmse_golden = compute_rela_errors(
        golden_dst_flat, golden_dst_upgrade_flat, dst_mask_golden
    )

    small_data_pass = compare_small_data(
        result_dst_flat, golden_dst_flat, golden_dst_upgrade_flat, dst_mask
    )

    result_scale_flat = result_scale.flatten()
    golden_scale_flat = golden_scale.numpy().flatten()
    golden_scale_upgrade_flat = golden_scale_upgrade.numpy().flatten()
    mare_scale, mere_scale, rmse_scale = compute_rela_errors(
        result_scale_flat, golden_scale_upgrade_flat, None
    )
    mare_scale_golden, mere_scale_golden, rmse_scale_golden = compute_rela_errors(
        golden_scale_flat, golden_scale_upgrade_flat, None
    )

    print("------ 综合精度指标 ------")
    print(f"dst: npu mare={mare_dst:.4f}, golden mare={mare_golden:.6f}")
    print(f"dst: npu mere={mere_dst:.4f}, golden mere={mere_golden:.6f}")
    print(f"dst: npu rmse={rmse_dst:.4f}, golden rmse={rmse_golden:.6f}")
    print(f"scale: npu mare={mare_scale:.4f}, golden mare={mare_scale_golden:.6f}")
    print(f"scale: npu mere={mere_scale:.4f}, golden mere={mere_scale_golden:.6f}")
    print(f"scale: npu rmse={rmse_scale:.4f}, golden rmse={rmse_scale_golden:.6f}")

    print("------ 开始比较 ------")
    mare_info_dst = compare_mare(mare_dst, mare_golden, "float8_e4m3fn")
    mere_info_dst = compare_mere(mere_dst, mere_golden, "float8_e4m3fn")
    rmse_info_dst = compare_rmse(rmse_dst, rmse_golden, "float8_e4m3fn")
    mare_info_scale = compare_mare(mare_scale, mare_scale_golden, dtype)
    mere_info_scale = compare_mere(mere_scale, mere_scale_golden, dtype)
    rmse_info_scale = compare_rmse(rmse_scale, rmse_scale_golden, dtype)

    res = (
        "Compare success"
        if (
            mare_info_dst
            & mere_info_dst
            & rmse_info_dst
            & small_data_pass
            & mare_info_scale
            & mere_info_scale
            & rmse_info_scale
        )
        else "Compare false"
    )
    print(f"精度指标比较结果：{res}")
