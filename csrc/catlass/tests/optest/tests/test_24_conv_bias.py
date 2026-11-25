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
import torch.nn.functional as F
from conv_common import (
    C0,
    coidkhkw_to_kdc1khkwn1n0c0,
    ncdhw_to_ndc1hwc0,
    ndc1hwc0_to_ncdhw,
)

import torch_catlass
from common import only_on_2201
from torch_catlass.ops.conv_bias import conv_output_numel


@only_on_2201
@pytest.mark.parametrize(
    "batch,cin,di,hi,wi,cout,kd,kh,kw,sd,sh,sw,dd,dh,dw,pd,ph,pw",
    [
        (1, 16, 1, 32, 48, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
        (2, 32, 1, 14, 14, 32, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
        (2, 16, 3, 36, 22, 48, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 0),
        (2, 16, 4, 64, 28, 16, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 0),
    ],
)
def test_conv_bias(
    batch,
    cin,
    di,
    hi,
    wi,
    cout,
    kd,
    kh,
    kw,
    sd,
    sh,
    sw,
    dd,
    dh,
    dw,
    pd,
    ph,
    pw,
):
    """Compare catlass conv bias against torch.conv3d (FP16 only)."""
    torch.manual_seed(1)

    fmap_t = torch.randn(batch, cin, di, hi, wi, dtype=torch.float32)
    weight_t = torch.randn(cout, cin, kd, kh, kw, dtype=torch.float32)
    bias_t = torch.randn(cout, dtype=torch.float32)

    golden = F.conv3d(
        fmap_t,
        weight_t,
        bias_t,
        stride=(sd, sh, sw),
        padding=(pd, ph, pw),
        dilation=(dd, dh, dw),
    )

    cin1 = (cin + C0 - 1) // C0

    fmap_cat = ncdhw_to_ndc1hwc0(fmap_t.half())
    weight_cat = coidkhkw_to_kdc1khkwn1n0c0(weight_t.half())
    bias_cat = bias_t.half()

    fmap_npu = fmap_cat.npu()
    weight_npu = weight_cat.npu()
    bias_npu = bias_cat.npu()

    out_numel = conv_output_numel(
        batch,
        di,
        cin1,
        hi,
        wi,
        C0,
        kd,
        kh,
        kw,
        cout,
        sd,
        sh,
        sw,
        dd,
        dh,
        dw,
        pd,
        ph,
        pw,
    )

    result = torch_catlass.conv_bias(
        fmap_npu,
        weight_npu,
        bias_npu,
        [batch, di, cin1, hi, wi, C0],
        [kd, kh, kw, cout],
        [sd, sh, sw],
        [pd, ph, pw],
        [dd, dh, dw],
        out_numel,
    )

    do = (di + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    ho = (hi + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    result_nchw = ndc1hwc0_to_ncdhw(result.cpu(), batch, do, cout, ho, wo)
    golden_half = golden.half()

    assert result_nchw.shape == golden_half.shape, f"{result_nchw.shape} vs {golden_half.shape}"
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    diff = (result_nchw.float() - golden_half.float()).abs().max().item()
    assert torch.allclose(result_nchw.float(), golden_half.float(), rtol=1e-2, atol=1e-2), (
        f"max diff = {diff}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
