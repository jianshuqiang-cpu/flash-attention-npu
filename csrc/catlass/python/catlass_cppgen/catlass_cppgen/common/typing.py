# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Union
import torch
import numpy as np

if TYPE_CHECKING:
    from catlass_cppgen.common.op_tensor import OpTensor

    _SupportedTensor = Union[torch.Tensor, np.ndarray, OpTensor]
else:
    _SupportedTensor = Union[torch.Tensor, np.ndarray]

"""
类型预留
"""

SupportedTensor = _SupportedTensor
SupportedDataType = Union[torch.dtype, np.dtype]


class GM_ADDR(ctypes.c_void_p):
    pass


# 未实现
class EpilogueParams(ctypes.c_void_p):
    pass
