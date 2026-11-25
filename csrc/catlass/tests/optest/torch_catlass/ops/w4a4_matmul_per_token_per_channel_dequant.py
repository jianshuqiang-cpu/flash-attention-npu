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

def w4a4_matmul_per_token_per_channel_dequant(
    mat1: Tensor, mat2: Tensor, scale: Tensor, per_token_scale: Tensor,
    outDType: str | torch.dtype = torch.bfloat16,
    transA: bool = False, transB: bool = False) -> Tensor:
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return torch.ops.catlass.w4a4_matmul_per_token_per_channel_dequant(
        mat1, mat2, scale, per_token_scale, outDType, transA, transB)
