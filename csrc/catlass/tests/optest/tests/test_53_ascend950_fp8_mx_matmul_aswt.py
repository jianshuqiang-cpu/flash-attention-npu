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

from mx_golden import prepare_fp8_mx_inputs


from common import only_on_3510


@pytest.mark.parametrize(
    "trans_a,trans_b",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=["nn", "nt", "tn", "tt"],
)
@only_on_3510
def test_ascend950_fp8_mx_matmul_aswt(trans_a, trans_b):
    """Compare CATLASS MX FP8 matmul (ASWT) against dequant reference for all transpose pairs."""
    m, n, k = 256, 512, 1024
    a, b, a_scale, b_scale, expected = prepare_fp8_mx_inputs(
        m, n, k, device="npu", trans_a=trans_a, trans_b=trans_b
    )

    result = torch_catlass.ascend950_fp8_mx_matmul_aswt(
        a, b, a_scale, b_scale, transA=trans_a, transB=trans_b
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(result.cpu(), expected, rtol=rtol, atol=atol), (
        f"trans_a={trans_a}, trans_b={trans_b}: max diff = "
        f"{(result.cpu() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
