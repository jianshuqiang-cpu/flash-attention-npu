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
from mx_golden import prepare_dual_level_quant_mx_batch_inputs

def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))

pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 63_ascend950_dual_level_quant_mx_batch_matmul requires Ascend 950 NPU",
)

def test_ascend950_dual_level_quant_mx_batch_matmul():
    """Compare CATLASS dual-level quant MX batch matmul against quantized reference."""
    import torch_catlass

    batch, m, n, k = 1, 256, 512, 1024
    a, b, expected = prepare_dual_level_quant_mx_batch_inputs(batch, m, n, k, device="npu")

    result, scale_a1, scale_a2, scale_b1, scale_b2 = (
        torch_catlass.ascend950_dual_level_quant_mx_batch_matmul(a, b)
    )

    scale1_k = (k + 511) // 512
    mx_scale_k = (k + 31) // 32
    mx_scale_aligned_k = ((mx_scale_k + 1) // 2) * 2

    assert result.shape == (batch, m, n)
    assert scale_a1.shape == (batch, m, scale1_k)
    assert scale_a2.shape == (batch, m, mx_scale_aligned_k)
    assert scale_b1.shape == (batch, n, scale1_k)
    assert scale_b2.shape == (batch, n, mx_scale_aligned_k)

    assert result.dtype == torch.bfloat16
    assert scale_a1.dtype == torch.float32
    assert scale_a2.dtype == torch.float8_e8m0fnu
    assert scale_b1.dtype == torch.float32
    assert scale_b2.dtype == torch.float8_e8m0fnu

    for tensor in (result, scale_a1, scale_a2, scale_b1, scale_b2):
        assert tensor.device.type == "npu"

    rtol, atol = 1e-1, 1e-1
    assert torch.allclose(result.cpu().float(), expected, rtol=rtol, atol=atol), (
        f"max diff = {(result.cpu().float() - expected).abs().max().item()}"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])