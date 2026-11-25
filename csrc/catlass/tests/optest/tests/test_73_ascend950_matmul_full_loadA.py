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


def _is_ascend950() -> bool:
    if torch_npu.npu.device_count() <= 0:
        return False
    name = torch_npu.npu.get_device_name()
    return bool(re.search(r"Ascend950(PR|DT)", name, re.I))


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason="example 73_ascend950_matmul_full_loadA requires Ascend 950 NPU",
)


def test_ascend950_matmul_full_loadA():
    """Compare CATLASS Ascend950 full-loadA matmul TLA against torch.matmul."""
    m, n, k = 512, 512, 256
    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")

    result = torch_catlass.ascend950_matmul_full_loadA(a, b, "float16", False, False, False, False)
    expected = torch.matmul(a, b)

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
