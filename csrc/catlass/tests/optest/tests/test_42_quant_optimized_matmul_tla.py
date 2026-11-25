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


from common import only_on_2201


@only_on_2201
def test_quant_optimized_matmul_tla():
    """Compare the CATLASS quant optimized matmul (TLA) wrapper against a reference computation."""
    m, n, k = 256, 512, 1024

    a = torch.randint(-5, 5, (m, k), dtype=torch.int8, device="npu")
    b = torch.randint(-5, 5, (n, k), dtype=torch.int8, device="npu")
    scale = torch.rand(1, n, dtype=torch.float32, device="npu")
    per_token_scale = torch.rand(1, m, dtype=torch.float32, device="npu")

    result = torch_catlass.quant_optimized_matmul_tla(
        a, b, scale, per_token_scale, "bfloat16", transA=False, transB=True
    )

    a_f32 = a.float()
    b_t = b.t().float()
    scale_t = scale.squeeze(0)
    per_token_scale_t = per_token_scale.squeeze(0)
    expected = per_token_scale_t.unsqueeze(1) * (a_f32 @ b_t) * scale_t.unsqueeze(0)
    expected = expected.to(torch.bfloat16).npu()

    assert result.shape == (m, n)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "npu"

    rtol = 1e-1
    atol = 1e-1
    assert torch.allclose(result.float(), expected.float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
