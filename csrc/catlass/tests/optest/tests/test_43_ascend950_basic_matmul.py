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


from common import only_on_3510


@only_on_3510
def test_ascend950_basic_matmul():
    """Compare CATLASS Ascend950 basic matmul TLA against torch.matmul."""
    m, n, k = 256, 512, 1024
    a = torch.randn(m, k, dtype=torch.float32, device="npu")
    b = torch.randn(k, n, dtype=torch.float32, device="npu")

    result = torch_catlass.ascend950_basic_matmul(a, b, "float32", False, False, False, False)
    expected = torch.matmul(a, b)

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
