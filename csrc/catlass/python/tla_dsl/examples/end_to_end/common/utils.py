# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch  # noqa: F401
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Helper functions below
# ---------------------------------------------------------------------------


def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Get launch ``block_num``.

    Non-``-1`` uses the host argument. ``-1`` means full-device launch:
    pure vector → ``vector_core_num`` (AIV); cube/mix → ``cube_core_num`` (AIC).
    """
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


def create_tla_tensor(buf, layout: str):
    tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
    return from_dlpack(buf.contiguous(), layout_tag=tag).mark_layout_dynamic()


def to_hf32(
    x: torch.Tensor,
    hf32_mode: tla.params.HF32Mode,
) -> torch.Tensor:
    """Simulate HF32 rounding mode on f32 values.

    HF32 keeps the FP32 sign and 8-bit exponent, and reduces the mantissa to
    11 significant bits (10 explicit mantissa bits, close to FP16):

    - ``HF32_NEAREST_ZERO`` rounds to nearest, ties toward zero.
    - ``HF32_NEAREST_EVEN`` rounds them to nearest-even.
    """
    if not isinstance(hf32_mode, tla.params.HF32Mode):
        raise TypeError(
            f"hf32_mode must be a tla.params.HF32Mode, got {type(hf32_mode).__name__}"
        )
    x = x.float()
    if hf32_mode == tla.params.HF32Mode.HF32_DISABLE:
        return x

    bits = x.contiguous().view(torch.int32)
    if hf32_mode == tla.params.HF32Mode.HF32_NEAREST_ZERO:
        rounded_bits = (bits + 0x0FFF) & ~0x1FFF
    elif hf32_mode == tla.params.HF32Mode.HF32_NEAREST_EVEN:
        lsb = (bits >> 13) & 1
        rounded_bits = (bits + 0x0FFF + lsb) & ~0x1FFF
    else:
        raise ValueError(f"Unsupported HF32 mode: {hf32_mode!r}")

    return rounded_bits.view(torch.float32)
