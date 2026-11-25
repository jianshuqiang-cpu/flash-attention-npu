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


def matrix_inverse(A: Tensor) -> Tensor:
    """Compute the inverse of a square matrix on NPU.

    Source: example 78_matrix_inverse.

    Uses LU decomposition with partial pivoting to compute :math:`A^{-1}`,
    satisfying :math:`A \\times A^{-1} = I`.

    Args:
        A: Square matrix of shape ``(N, N)``, dtype ``float32``.
            Must be non-singular (invertible).

    Returns:
        Inverse matrix :math:`A^{-1}` with the same shape and dtype as ``A``.
    """
    return torch.ops.catlass.matrix_inverse(A)
