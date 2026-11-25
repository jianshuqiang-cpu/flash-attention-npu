# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor


class EvgPostprocessMode(str, Enum):
    """EVG matmul post-processing mode (example 64_ascend950_matmul_evg)."""

    ADD = "add"
    ADD_UB = "add_ub"
    BIAS = "bias"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"
    SILU = "silu"
    TANH = "tanh"


def _resolve_out_dtype(outDType: str | torch.dtype) -> torch.dtype:
    if isinstance(outDType, str):
        dtype_lower = outDType.lower()
        outDType = getattr(torch, dtype_lower, None)
    if outDType is None:
        raise ValueError(f"{outDType} is not a data type of torch")
    return outDType


def _resolve_mode(mode: EvgPostprocessMode | str) -> str:
    if isinstance(mode, EvgPostprocessMode):
        return mode.value
    if mode in {m.value for m in EvgPostprocessMode}:
        return mode
    supported = ", ".join(m.value for m in EvgPostprocessMode)
    raise ValueError(f"unsupported EVG postprocess mode {mode!r}; choose from: {supported}")


def _empty_extra(device: torch.device) -> Tensor:
    return torch.empty(0, device=device)


def ascend950_matmul_evg(
    mat1: Tensor,
    mat2: Tensor,
    *,
    mode: EvgPostprocessMode | str = EvgPostprocessMode.ADD,
    extra: Tensor | None = None,
    negative_slope: float = 0.1,
    outDType: str | torch.dtype = torch.float32,
    transA: bool = False,
    transB: bool = False,
    formatA: bool = False,
    formatB: bool = False,
) -> Tensor:
    """Ascend950 EVG matmul with selectable post-processing.

    Computes ``mat1 @ mat2`` fused with a post-processing op selected by ``mode``.
    Dispatches to ``torch.ops.catlass.matmul_evg`` with ``evgType=mode``.

    Source: examples/64_ascend950_matmul_evg/.
    """
    evg_type = _resolve_mode(mode)
    resolved_out_dtype = _resolve_out_dtype(outDType)

    if extra is None:
        extra = _empty_extra(mat1.device)

    return torch.ops.catlass.matmul_evg(
        mat1,
        mat2,
        extra,
        resolved_out_dtype,
        evg_type,
        negative_slope,
        transA,
        transB,
        formatA,
        formatB,
    )
