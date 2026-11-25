# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import torch
from torch import Tensor


def conv_bias(
    fmap: Tensor,
    weight: Tensor,
    bias: Tensor,
    fmap_related: list,
    filter_related: list,
    stride_list: list,
    pad_list: list,
    dilation_list: list,
    out_numel: int,
) -> Tensor:
    """Run CATLASS 3D conv bias (fixed FP16) on NPU tensors in catlass private layouts.

    Source: example 24_conv_bias.

    Args:
        fmap: Feature-map in NDC1HWC0 layout ``(batch*di*cin1*hi*wi*cin0,)``, float16.
        weight: Filter in KDC1KHKWN1N0C0 layout ``(kd*cin1*kh*kw, n1, n0, cin0)``, float16.
        bias: Bias in ND layout ``(cout,)``, float16.
        fmap_related: ``[batch, di, cin1, hi, wi, cin0]``.
        filter_related: ``[kd, kh, kw, cout]``.
        stride_list: ``[sd, sH, sW]``.
        pad_list: ``[pD, pH, pW]``.
        dilation_list: ``[dD, dH, dW]``.
        out_numel: Number of elements in output tensor (pre-computed).

    Returns:
        Output in NDC1HWC0 layout ``(n*do*cout1*ho*wo*cout0,)``, float16.
    """
    return torch.ops.catlass.conv_bias(
        fmap,
        weight,
        bias,
        fmap_related,
        filter_related,
        stride_list,
        pad_list,
        dilation_list,
        out_numel,
    )


def conv_output_numel(
    batch, di, cin1, hi, wi, cin0, kd, kh, kw, cout, sd, sh, sw, dd, dh, dw, pd, ph, pw
):
    """Compute output element count in NDC1HWC0 layout."""
    cout1 = (cout + cin0 - 1) // cin0
    d_out = (di + 2 * pd - dd * (kd - 1) - 1) // sd + 1
    h_out = (hi + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    w_out = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return batch * d_out * cout1 * h_out * w_out * cin0
