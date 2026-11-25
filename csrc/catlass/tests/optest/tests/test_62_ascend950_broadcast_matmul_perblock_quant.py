# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import pytest
import torch
import torch_npu
import numpy as np
from ml_dtypes import float8_e4m3fn


from common import only_on_3510


E4M3_MAX = 448.0
E4M3_MIN = -448.0


def _build_e4m3_lut():
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            value = 0.0
        elif val == 127:
            value = sign * E4M3_MAX
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exp - 7)) if exp != 0 else (mantissa / 8.0) * (2.0 ** (1 - 7))
            value = sign * value
        values.append(max(E4M3_MIN, min(E4M3_MAX, value)))
    return torch.tensor(values, dtype=torch.float32)


E4M3_LUT = _build_e4m3_lut()


def _quantize_to_e4m3(data):
    clamped = torch.clamp(data, E4M3_MIN, E4M3_MAX)
    dist = torch.abs(clamped.unsqueeze(-1) - E4M3_LUT)
    return torch.argmin(dist, dim=-1).to(torch.uint8)


def _compute_scale_and_quantize(matrix):
    abs_max = torch.amax(torch.abs(matrix))
    scale = torch.clamp(abs_max, min=1e-12) / E4M3_MAX
    return _quantize_to_e4m3(matrix / scale), scale.item()


def _gen_data(batch_count, m, n, k, a_bf16, b_bf16):
    a_fp32, b_fp32 = a_bf16.float(), b_bf16.float()
    c_bf16 = torch.matmul(a_bf16, b_bf16)
    c_fp32 = torch.matmul(a_fp32, b_fp32)

    golden_dst, golden_scale = torch.zeros(batch_count, m, n, dtype=torch.uint8), torch.zeros(batch_count, dtype=torch.float32)
    golden_dst_upgrade, golden_scale_upgrade = torch.zeros(batch_count, m, n, dtype=torch.uint8), torch.zeros(batch_count, dtype=torch.float32)

    for b_idx in range(batch_count):
        golden_dst[b_idx], golden_scale[b_idx] = _compute_scale_and_quantize(c_bf16[b_idx].float())
        golden_dst_upgrade[b_idx], golden_scale_upgrade[b_idx] = _compute_scale_and_quantize(c_fp32[b_idx])

    return golden_dst, golden_scale, golden_dst_upgrade, golden_scale_upgrade


def _compute_rela_errors(result, golden, mask=None):
    relative_errors = np.abs((result - golden) / (golden + 1e-7))
    mse = np.sqrt(np.mean((result - golden) ** 2))
    if mask is not None and np.any(mask):
        return np.max(relative_errors[mask]), np.mean(relative_errors[mask]), mse
    return np.max(relative_errors), np.mean(relative_errors), mse


def _compare_mare(mare_npu, mare_upgrade, dtype):
    err = 2 ** (-6) if dtype == "float8_e4m3fn" else 2 ** (-11)
    return (mare_npu / max(mare_upgrade, err)) < 10


def _compare_mere(mere_npu, mere_upgrade, dtype):
    err = 2 ** (-6) if dtype == "float8_e4m3fn" else 2 ** (-11)
    return (mere_npu / max(mere_upgrade, err)) < 2


def _compare_rmse(rmse_npu, rmse_upgrade, dtype):
    err = 2 ** (-2) if dtype == "float8_e4m3fn" else 2 ** (-6)
    return (rmse_npu / max(rmse_upgrade, err)) < 2


def _compare_small_data(result_dst_flat, golden_dst_flat, golden_dst_upgrade_flat, dst_mask):
    ERR_THRESHOLD = 2 ** (-6)
    small_dst_mask = ~dst_mask
    error_count_golden_small = np.sum(small_dst_mask & (np.abs(golden_dst_flat - golden_dst_upgrade_flat) > ERR_THRESHOLD))
    error_count_result_small = np.sum(small_dst_mask & (np.abs(result_dst_flat - golden_dst_upgrade_flat) > ERR_THRESHOLD))
    return (error_count_result_small / max(error_count_golden_small, 1)) < 2


@only_on_3510
def test_broadcast_matmul_perblock_quant():
    """Compare the CATLASS broadcast matmul per-block quant wrapper against a reference computation."""
    import torch_catlass

    batch_count = 1
    m, k, n = 128, 128, 128

    a = torch.randn(batch_count, m, k, dtype=torch.bfloat16, device="npu")
    b = torch.randn(k, n, dtype=torch.bfloat16, device="npu")

    dst, scale = torch_catlass.broadcast_matmul_perblock_quant(a, b)

    assert dst.shape == (batch_count, m, n)
    assert scale.shape == (batch_count,)
    assert dst.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert dst.device.type == "npu"
    assert scale.device.type == "npu"


@only_on_3510
def test_broadcast_matmul_perblock_quant_correctness():
    """Verify the broadcast matmul per-block quant output against golden reference with comprehensive metrics."""
    import torch_catlass

    batch_count, m, k, n = 5920, 128, 128, 128

    a = torch.randn(batch_count, m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)

    dst, scale = torch_catlass.broadcast_matmul_perblock_quant(a.npu(), b.npu())

    golden_dst, golden_scale, golden_dst_upgrade, golden_scale_upgrade = _gen_data(batch_count, m, n, k, a, b)

    dst_np = dst.cpu().view(torch.uint8).numpy().flatten().view(float8_e4m3fn).astype(np.float32)
    golden_dst_flat = golden_dst.numpy().flatten().view(float8_e4m3fn).astype(np.float32)
    golden_dst_upgrade_flat = golden_dst_upgrade.numpy().flatten().view(float8_e4m3fn).astype(np.float32)

    SMALL_THRESHOLD = 2 ** (-6)
    dst_mask = np.minimum(np.abs(dst_np), np.abs(golden_dst_upgrade_flat)) > SMALL_THRESHOLD
    mare_dst, mere_dst, rmse_dst = _compute_rela_errors(dst_np, golden_dst_upgrade_flat, dst_mask)
    dst_mask_golden = np.minimum(np.abs(golden_dst_flat), np.abs(golden_dst_upgrade_flat)) > SMALL_THRESHOLD
    mare_golden, mere_golden, rmse_golden = _compute_rela_errors(golden_dst_flat, golden_dst_upgrade_flat, dst_mask_golden)

    scale_np = scale.cpu().numpy().flatten()
    golden_scale_flat = golden_scale.numpy().flatten()
    golden_scale_upgrade_flat = golden_scale_upgrade.numpy().flatten()
    mare_scale, mere_scale, rmse_scale = _compute_rela_errors(scale_np, golden_scale_upgrade_flat, None)
    mare_scale_golden, mere_scale_golden, rmse_scale_golden = _compute_rela_errors(golden_scale_flat, golden_scale_upgrade_flat, None)

    assert _compare_mare(mare_dst, mare_golden, "float8_e4m3fn"), f"MARE dst failed: npu={mare_dst}, golden={mare_golden}"
    assert _compare_mere(mere_dst, mere_golden, "float8_e4m3fn"), f"MERE dst failed: npu={mere_dst}, golden={mere_golden}"
    assert _compare_rmse(rmse_dst, rmse_golden, "float8_e4m3fn"), f"RMSE dst failed: npu={rmse_dst}, golden={rmse_golden}"
    assert _compare_small_data(dst_np, golden_dst_flat, golden_dst_upgrade_flat, dst_mask), "Small data comparison failed"
    assert _compare_mare(mare_scale, mare_scale_golden, "bfloat16"), f"MARE scale failed: npu={mare_scale}, golden={mare_scale_golden}"
    assert _compare_mere(mere_scale, mere_scale_golden, "bfloat16"), f"MERE scale failed: npu={mere_scale}, golden={mere_scale_golden}"
    assert _compare_rmse(rmse_scale, rmse_scale_golden, "bfloat16"), f"RMSE scale failed: npu={rmse_scale}, golden={rmse_scale_golden}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
