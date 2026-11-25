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


def ascend950_svd_quant_matmul(
    x: Tensor,
    svd1: Tensor,
    svd2: Tensor,
    w: Tensor,
    w_scale: Tensor,
    smooth_scale: Tensor,
    qmax: float = 8.0,
) -> Tensor:
    """Run CATLASS Ascend950 SvdQuant matmul on NPU tensors.

    Source: example 61_ascend950_svd_quant_matmul.

    Args:
        x: Input activation, shape ``(M, K)``, dtype fp16.
        svd1: First SVD factor, shape ``(K, R)``, dtype fp16, ColumnMajor layout.
        svd2: Second SVD factor, shape ``(R, N)``, dtype fp16, ColumnMajor layout.
        w: Packed FP4 residual weight as 1-D int8 bytes (ColumnMajor ``K x N`` layout).
        w_scale: MX scale for W (float8_e8m0fnu), shape ``(N, mxScaleAlignedK/2, 2)``.
        smooth_scale: Smooth scale vector, shape ``(K,)``, dtype fp16.
        qmax: SmoothQuant qmax (must match kernel configuration).

    Returns:
        FP16 output tensor with shape ``(M, N)``.
    """
    return torch.ops.catlass.ascend950_svd_quant_matmul(
        x, svd1, svd2, w, w_scale, smooth_scale, qmax
    )
