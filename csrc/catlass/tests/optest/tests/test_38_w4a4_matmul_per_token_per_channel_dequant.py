# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import pytest
import numpy as np
import torch
import torch_npu
import torch_catlass

from common import only_on_2201


def _pack_int4_np(data: np.ndarray) -> np.ndarray:
    """Pack int8 values into int4 format, matching examples/38 gen_data.py logic."""
    data_i8 = data.astype(np.int8)
    data_i8[data_i8 < 0] += 16
    packed = ((data_i8[..., 1::2] << 4) | (data_i8[..., ::2] & 0x0F)).astype(np.int8)
    return packed


def _pack_int4_zN_np(weight: np.ndarray) -> np.ndarray:
    """Pack and zN-reorder int4 weight matrix."""
    k, n_phys = weight.shape
    n_log = n_phys
    packed = _pack_int4_np(weight)
    packed = packed.reshape(k, n_log // 2)
    packed = packed.reshape(k // 16, 16, n_log // 64, 32)
    packed = packed.transpose(2, 0, 1, 3).reshape(k, n_log // 2)
    return packed


def _pack_scale_to_uint64_np(scale: np.ndarray) -> np.ndarray:
    """Pack float32 scale into uint64 format as expected by the kernel."""
    n = scale.shape[0]
    scale_f32 = scale.reshape(1, n).astype(np.float32)
    scale_u32 = scale_f32.view(np.uint32)
    result = np.zeros((1, 2 * n), dtype=np.uint32)
    result[..., ::2] = scale_u32
    return result.view(np.uint64).squeeze()


@only_on_2201
def test_w4a4_matmul_per_token_per_channel_dequant():
    m, n_log, k_log = 128, 256, 512

    a_i4 = np.random.randint(-8, 8, (m, k_log)).astype(np.int8)
    b_i4 = np.random.randint(-8, 8, (k_log, n_log)).astype(np.int8)
    scale_f32 = np.random.normal(0, 0.01, (1, n_log)).astype(np.float32)
    per_token_scale_f32 = np.random.normal(0, 0.01, (m, 1)).astype(np.float32)

    a_packed = torch.from_numpy(_pack_int4_np(a_i4).copy()).npu()
    b_packed = torch.from_numpy(_pack_int4_zN_np(b_i4).copy()).npu()
    scale_packed = torch.from_numpy(_pack_scale_to_uint64_np(scale_f32.flatten()).copy().view(np.int64)).npu()
    per_token_scale = torch.from_numpy(per_token_scale_f32.squeeze().copy()).npu()

    result = torch_catlass.w4a4_matmul_per_token_per_channel_dequant(
        a_packed, b_packed, scale_packed, per_token_scale, torch.bfloat16,
        transA=False, transB=False,
    )

    a_f32 = torch.from_numpy(a_i4.astype(np.float32))
    b_f32 = torch.from_numpy(b_i4.astype(np.float32))
    mm = torch.matmul(a_f32, b_f32).to(torch.float16)
    mm_f32 = mm.float() * torch.from_numpy(scale_f32.astype(np.float32))
    mm_f32 = mm_f32 * torch.from_numpy(per_token_scale_f32.astype(np.float32))
    expected = mm_f32.to(torch.bfloat16).npu()

    assert result.shape == (m, n_log)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"

    rtol = 1e-1
    atol = 1e-1
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
