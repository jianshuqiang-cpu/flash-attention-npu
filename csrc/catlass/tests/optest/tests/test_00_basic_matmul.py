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

import torch_catlass
from common import only_on_2201


@only_on_2201
def test_basic_matmul():
    """Compare the CATLASS basic matmul wrapper against ``torch.matmul``."""
    m, n, k = 16, 16, 16

    a = torch.randn(m, k, dtype=torch.float16, device="npu")
    b = torch.randn(k, n, dtype=torch.float16, device="npu")

    print(f"\nInput A shape: {a.shape}, dtype: {a.dtype}")
    print(f"Input B shape: {b.shape}, dtype: {b.dtype}")
    print(f"Input A sample: {a[0, :5]}")
    print(f"Input B sample: {b[0, :5]}")

    torch_catlass.clear_jit_cache()
    result = torch_catlass.basic_matmul(a, b, "float16", False, False, False, False)
    expected = torch.matmul(a, b)

    print(f"Shape: {result.shape}, dtype: {result.dtype}")

    assert result.shape == (m, n)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
