# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
# OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch
import torch_catlass
import torch_npu  # noqa: F401  required for NPU tensor dispatch
from common import only_on_2201


@only_on_2201
def test_planar_complex_matmul():
    m, n, k = 128, 256, 256
    torch.manual_seed(42)

    a_real = (torch.rand((m, k), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    a_imag = (torch.rand((m, k), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    b_real = (torch.rand((k, n), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)
    b_imag = (torch.rand((k, n), dtype=torch.float32, device="npu") * 10 - 5).to(torch.float16)

    result_real, result_imag = torch_catlass.planar_complex_matmul(a_real, a_imag, b_real, b_imag)
    expected_real = torch.matmul(a_real.float(), b_real.float()) - torch.matmul(
        a_imag.float(), b_imag.float()
    )
    expected_imag = torch.matmul(a_real.float(), b_imag.float()) + torch.matmul(
        a_imag.float(), b_real.float()
    )

    assert result_real.shape == (m, n)
    assert result_imag.shape == (m, n)
    assert result_real.dtype == torch.float32
    assert result_imag.dtype == torch.float32
    assert result_real.device.type == "npu"
    assert result_imag.device.type == "npu"

    rtol = 5e-2
    atol = 5e-2
    assert torch.allclose(result_real, expected_real, rtol=rtol, atol=atol), (
        f"real result max diff = {(result_real - expected_real).abs().max().item()}"
    )
    assert torch.allclose(result_imag, expected_imag, rtol=rtol, atol=atol), (
        f"imag result max diff = {(result_imag - expected_imag).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
