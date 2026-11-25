# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import re

import pytest
import torch
import torch_npu
import torch_catlass

from mx_golden import prepare_grouped_mx_swiglu_quant_inputs

def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))

pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant requires Ascend 950 NPU",
)

@pytest.mark.parametrize(
    "group_count,m,n,k",
    [
        (4, 256, 512, 1024),
    ],
    ids=["g4_m256_n512_k1024"],
)
def test_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant(group_count, m, n, k):
    """Compare CATLASS Grouped MX FP8 matmul + SwiGLU + MX quant against golden reference."""
    (a_fp8, b_fp8_trans, a_scale, b_scale_trans, group_list,
    expected_q, expected_q_scale) = prepare_grouped_mx_swiglu_quant_inputs(
        m, n, k, group_count, device="npu"
    )
    results = torch_catlass.ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant(
        a_fp8, b_fp8_trans, a_scale, b_scale_trans, group_list, transB=True,
    )

    q_result = results[0]
    q_scale_result = results[1]

    n_half = n // 2
    q_scale_cols = (n_half + 31) // 32

    assert q_result.shape == (m, n_half), f"Q shape: expected {(m, n_half)}, got {q_result.shape}"
    assert q_result.dtype == torch.float8_e4m3fn, f"Q dtype: expected float8_e4m3fn, got {q_result.dtype}"
    assert q_result.device.type == "npu"

    assert q_scale_result.shape == (m, q_scale_cols), (
        f"Q_scale shape: expected {(m, q_scale_cols)}, got {q_scale_result.shape}"
    )
    assert q_scale_result.dtype == torch.float8_e8m0fnu, (
        f"Q_scale dtype: expected float8_e8m0fnu, got {q_scale_result.dtype}"
    )
    assert q_scale_result.device.type == "npu"

    q_cpu = q_result.cpu()
    q_scale_cpu = q_scale_result.cpu()

    q_match = (q_cpu.view(torch.uint8) == expected_q.view(torch.uint8)).float().mean().item()
    q_scale_match = (q_scale_cpu.view(torch.uint8) == expected_q_scale.view(torch.uint8)).float().mean().item()

    assert q_match > 0.99, f"Q byte match rate: {q_match:.4f} (expected > 0.99)"
    assert q_scale_match > 0.99, f"Q_scale byte match rate: {q_scale_match:.4f} (expected > 0.99)"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])