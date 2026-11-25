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
    cihw_to_ci1khkwcoci0,
    conv_can_implement,
    get_howo,
    nc1hwc0_to_nchw,
    nchw_to_nc1hwc0,
)

import torch_catlass
from common import only_on_3510


@only_on_3510
def test_ascend950_basic_conv2d_tla():
    torch.manual_seed(2)
    C0 = 16
    N, C, H, W = 1, 16, 8, 8
    OC, KH, KW = 32, 3, 3
    stride = [1, 1]
    padding = [1, 2, 0, 1]
    dilation = [1, 1]

    fmap_cpu = torch.randn(N, C, H, W, dtype=torch.float16)
    filter_cpu = torch.randn(OC, C, KH, KW, dtype=torch.float16)

    fmap_nc1hwc0 = nchw_to_nc1hwc0(fmap_cpu)
    filter_ci1khkwcoci0 = cihw_to_ci1khkwcoci0(filter_cpu)

    fmap_npu = fmap_nc1hwc0.npu()
    filter_npu = filter_ci1khkwcoci0.npu()

    result = torch_catlass.ascend950_basic_conv2d_tla(
        fmap_npu,
        filter_npu,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    Ho, Wo = get_howo((H, W), (KH, KW), padding, dilation, stride)
    CO1 = (OC + C0 - 1) // C0
    expected_size = N * CO1 * Ho * Wo * C0

    assert result.numel() == expected_size
    assert result.dtype == torch.float16
    assert result.device.type == "npu"

    # F.pad order: (padLeft, padRight, padTop, padBottom) for last 2 dims
    fmap_padded = F.pad(fmap_cpu.float(), (padding[0], padding[1], padding[2], padding[3]))
    ref_output = F.conv2d(
        fmap_padded,
        filter_cpu.float(),
        stride=stride,
        padding=0,
        dilation=dilation,
    )

    result_nchw = nc1hwc0_to_nchw(result.cpu(), N, OC, Ho, Wo)

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(result_nchw.float(), ref_output, rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result_nchw.float() - ref_output).abs().max().item()}"
    )


@only_on_3510
def test_ascend950_basic_conv2d_tla_exceeds_cache():
    torch.manual_seed(2)
    N, C, H, W = 1, 16, 8, 8
    OC, KH, KW = 128, 5, 5
    stride = [1, 1]
    padding = [1, 2, 0, 1]
    dilation = [1, 1]
    if conv_can_implement((KH, KW), dilation, stride, l0c_size=256 * 1024):
        pytest.skip(
            "This case aims to test the case that the conv2d problem exceeds the on-chip buffers."
        )
    with pytest.raises(RuntimeError, match=r"Conv2d op cannot be implemented.*"):
        fmap_cpu = torch.randn(N, C, H, W, dtype=torch.float16)
        filter_cpu = torch.randn(OC, C, KH, KW, dtype=torch.float16)

        fmap_nc1hwc0 = nchw_to_nc1hwc0(fmap_cpu)
        filter_ci1khkwcoci0 = cihw_to_ci1khkwcoci0(filter_cpu)

        fmap_npu = fmap_nc1hwc0.npu()
        filter_npu = filter_ci1khkwcoci0.npu()

        _ = torch_catlass.ascend950_basic_conv2d_tla(
            fmap_npu,
            filter_npu,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
