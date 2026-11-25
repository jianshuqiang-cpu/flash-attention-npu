# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import overload

import torch

# HF32 mixed tolerance thresholds.
_HF32_RTOL = 2.0**-9
_HF32_ATOL = 2.0**-10
_HF32_REQUIRED_MATCHED_RATIO = 0.99
_HF32_MAX_ABS_ERROR_LIMIT = 1e-1


def tolerance(
    expected: torch.Tensor,
    k: int,
    *,
    bf16: bool,
    rtol: float | None = None,
    floor: float | None = None,
) -> torch.Tensor:
    """Return the element-wise pass threshold for the matmul precision standard."""
    if rtol is None:
        if bf16:
            rtol = (1.0 / 128.0) if k < 2048 else (1.0 / 64.0)
        else:
            rtol = (1.0 / 256.0) if k < 2048 else (1.0 / 128.0)
    if floor is None:
        floor = 1.0 / 256.0 if bf16 else 1.0
    return rtol * torch.maximum(torch.full_like(expected, floor), expected.abs())


def _compare_hf32(
    result: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    """Compare result against an HF32-semantic golden."""
    result = result.float()
    expected = expected.float()
    diff = (result - expected).abs()
    matched = diff <= _HF32_ATOL + _HF32_RTOL * expected.abs()
    matched_ratio = matched.float().mean().item()
    max_abs_error = diff.max().item()
    ulp = torch.finfo(expected.dtype).eps * (
        2.0 ** torch.floor(torch.log2(expected.abs().max()))
    )
    max_abs_error_limit = max(_HF32_MAX_ABS_ERROR_LIMIT, 32.0 * ulp.item())
    return bool(
        matched_ratio >= _HF32_REQUIRED_MATCHED_RATIO
        and max_abs_error <= max_abs_error_limit
    )


@overload
def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    k: int,
    *,
    rtol: float | None = None,
    floor: float | None = None,
) -> bool: ...


@overload
def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    *,
    enable_hf32: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> bool: ...


def compare(
    result: torch.Tensor,
    expected: torch.Tensor,
    k: int | None = None,
    *,
    rtol: float | None = None,
    floor: float | None = None,
    atol: float = 0.0,
    enable_hf32: bool = False,
) -> bool:
    """Compare ``result`` against ``expected`` with given threshold.

    Three call forms are supported:

    1. Accumulative precision standard (``k`` given), for matmul-like operators.
    2. Generic element-wise check.
    3. HF32 mixed tolerance (``enable_hf32=True``).
    """
    if enable_hf32:
        # Use mixed tolerance for HF32.
        return _compare_hf32(result, expected)
    if k is not None and isinstance(k, int):
        # Single precision standard.
        is_bf16 = result.dtype == torch.bfloat16
        result, expected = result.float(), expected.float()
        return bool(
            (
                (result - expected).abs()
                <= tolerance(expected, k, bf16=is_bf16, rtol=rtol, floor=floor)
            ).all()
        )
    if result.dtype != expected.dtype:
        raise TypeError("the data type between the golden and the result do not match")
    if result.dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Generic element-wise precision check.
        return bool(
            torch.isclose(
                result, expected, rtol=0.0 if rtol is None else rtol, atol=atol
            ).all()
        )
    return bool(result.eq(expected).all())
