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
import torch_catlass

from a8w4_golden import compare_a8w4_result, prepare_a8w4_mx_inputs
from common import only_on_3510


@only_on_3510
def test_ascend950_a8w4_mx_matmul():
    """Compare CATLASS A8W4 MX matmul against dequant reference."""
    m, n, k = 128, 128, 128
    a, b, a_scale, b_scale, expected = prepare_a8w4_mx_inputs(m, n, k, device="npu")

    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.int8
    assert a.shape == (m, k)
    assert b.numel() == (k * n + 1) // 2

    result = torch_catlass.ascend950_a8w4_mx_matmul(a, b, a_scale, b_scale)

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"

    torch_npu.npu.synchronize()
    result_cpu = result.cpu()
    assert torch.isfinite(result_cpu).all()

    passed, max_diff = compare_a8w4_result(result_cpu, expected, k)
    rtol = 1.0 / 128 if k >= 2048 else 1.0 / 256
    assert passed, (
        f"max diff = {max_diff}, rtol={rtol} "
        f"(CompareData: diff <= rtol * max(1, |expected|))"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
