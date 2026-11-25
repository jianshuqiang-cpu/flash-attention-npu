#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
# ----------------------------------------------------------------------------
"""Generate MX-FP8 input data and the FP32 golden reference for the
``53_ascend950_fp8_mx_matmul`` and related examples (e.g. ASWT sibling).

Use ``--data-root DIR`` to write ``DIR/data/``; if omitted, output goes next
to this script (``<script-dir>/data/``).

The implementation is fully vectorized, byte-identical to the previous
LUT-based reference, and ~20-150x faster on typical shapes.

Pipeline:

1. ``randn() * 10`` random FP32 matrices for A (``M,K``) and B (``K,N``).
2. Per-block max-abs along the K axis (block size 32) → E8M0 exponent
   computed from the FP32 biased exponent (bit-extraction). ``floor(log2(x))
   - emax`` is by definition equal to ``biased_exp(x) - 127 - emax`` for any
   positive finite FP32 ``x``, which makes this strictly equivalent to the
   reference ``int(math.floor(math.log2(x))) - emax``.
3. Divide by the (power-of-two) per-block scale; clamp to ``±fp8_max``.
4. Snap to the nearest FP8 grid point with the same semantics as the
   reference ``argmin`` over the 256-entry LUT (lowest-index tie breaking,
   i.e. round-half-toward-zero). Implemented with ``searchsorted`` against
   the positive half of the LUT plus a ``-0 → +0`` canonicalization to
   match the full-LUT argmin rule that puts ``+0`` (idx 0) ahead of ``-0``
   (idx 128) on ties.
5. Dequantize (lossless: cast back to FP32, multiply by the power-of-two
   scale) and run ``A_dequant @ B_dequant`` for the golden.

For development / one-off speed-ups where bit-identity to the reference is
not required, set ``GEN_DATA_QUANT_BACKEND=native`` to swap the snap step
for ``tensor.to(fp8_dtype)`` (round-half-to-even).
"""

import argparse
import os
from typing import Optional, Tuple

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_BLOCK_SIZE = 32
_EPSILON = 1e-12

# Quantization backend.
#   "lut"    – default. Vectorized argmin against the FP8 LUT via
#              ``searchsorted`` on the positive half plus lower-index tie
#              breaking. Bit-identical to the baseline (verified for B2 with
#              multiple seeds and for T1).
#   "native" – use ``tensor.to(fp8_dtype)`` (round-half-to-even). Faster but
#              differs from the baseline by 1 ULP on the rare scaled inputs
#              that fall exactly on a midpoint between two FP8 grid points.
#              Use only when bit-identity vs the original gen_data isn't
#              required.
# Override at runtime with the GEN_DATA_QUANT_BACKEND env var.
_QUANT_BACKEND = os.environ.get("GEN_DATA_QUANT_BACKEND", "lut").lower()
if _QUANT_BACKEND not in ("native", "lut"):
    raise ValueError(
        f"GEN_DATA_QUANT_BACKEND must be 'native' or 'lut' (got {_QUANT_BACKEND!r})"
    )

_FP8_FORMATS = {
    "E4M3": dict(
        torch_dtype=torch.float8_e4m3fn,
        emax=8,
        max_value=448.0,
        bias=7,
    ),
    "E5M2": dict(
        torch_dtype=torch.float8_e5m2,
        emax=15,
        max_value=57344.0,
        bias=15,
    ),
}


# --------------------------------------------------------------------------- #
# FP8 LUT construction (kept identical to the baseline so that argmin returns
# exactly the same indices on identical inputs).
# --------------------------------------------------------------------------- #
def _build_e4m3_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E4M3"]["bias"]
    fp8_max = _FP8_FORMATS["E4M3"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif val == 127:
            v = sign * fp8_max
        else:
            exp = (val >> 3) & 0x0F
            mantissa = val & 0x07
            if exp == 0:
                v = (mantissa / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 8.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


def _build_e5m2_lut() -> torch.Tensor:
    bias = _FP8_FORMATS["E5M2"]["bias"]
    fp8_max = _FP8_FORMATS["E5M2"]["max_value"]
    values = []
    for i in range(256):
        if i < 128:
            sign, val = 1, i
        else:
            sign, val = -1, i - 128
        if val == 0:
            v = 0.0
        elif 124 <= val <= 127:
            v = sign * fp8_max
        else:
            exp = (val >> 2) & 0x1F
            mantissa = val & 0x03
            if exp == 0:
                v = (mantissa / 4.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + mantissa / 4.0) * (2.0 ** (exp - bias))
            v *= sign
        v = max(min(v, fp8_max), -fp8_max)
        values.append(v)
    return torch.tensor(values, dtype=torch.float32)


_LUT_BUILDERS = {"E4M3": _build_e4m3_lut, "E5M2": _build_e5m2_lut}
_LUT_CACHE = {}
_LUT_POS_CACHE = {}


def _get_lut(format_name: str) -> torch.Tensor:
    if format_name not in _LUT_CACHE:
        _LUT_CACHE[format_name] = _LUT_BUILDERS[format_name]()
    return _LUT_CACHE[format_name]


def _get_lut_pos(format_name: str) -> torch.Tensor:
    """Return the positive half of the LUT (indices 0..127), sorted ascending.

    The full LUT is constructed as
        idx 0..127   – positive values, ascending by magnitude
        idx 128..255 – the same magnitudes negated (idx 128 = -0)
    so quantizing ``|x|`` against ``lut_pos`` reproduces the full-LUT
    ``argmin`` exactly when we then re-apply the sign and use lower-index
    tie-breaking. Verified ascending in the cache builder below.
    """
    if format_name not in _LUT_POS_CACHE:
        full = _get_lut(format_name)
        pos = full[:128].contiguous()
        # Sanity check on construction (no runtime cost after first call).
        diffs = pos[1:] - pos[:-1]
        if (diffs < 0).any():
            raise AssertionError(
                f"{format_name} positive LUT half is not non-decreasing"
            )
        _LUT_POS_CACHE[format_name] = pos
    return _LUT_POS_CACHE[format_name]


# --------------------------------------------------------------------------- #
# E8M0 scale exponent (vectorized exact replacement for
# MXFP8MatrixQuantizer._compute_scale_fp8_e8m0fnu).
# --------------------------------------------------------------------------- #
def _e8m0_exp(
    max_abs: torch.Tensor, emax: int, epsilon: float = _EPSILON
) -> torch.Tensor:
    """Return the per-block E8M0 exponent (``int32`` tensor, same shape as
    ``max_abs``). For each element:

        if max_abs < epsilon: exp = 0
        else: exp = clamp(floor(log2(max_abs)) - emax, -128, 127)

    ``floor(log2(x))`` for a positive finite FP32 ``x`` is exactly
    ``biased_exp(x) - 127``, so we extract it from the bit pattern.
    """
    assert max_abs.dtype == torch.float32, max_abs.dtype
    zero_mask = max_abs < epsilon
    safe = torch.where(zero_mask, torch.ones_like(max_abs), max_abs)
    # Reinterpret cast: same buffer, viewed as int32. .contiguous() is
    # required because .view(dtype) demands a contiguous tensor.
    bits = safe.contiguous().view(torch.int32)
    exp_bits = (bits >> 23) & 0xFF
    exp = exp_bits - 127 - emax
    exp = exp.clamp(-128, 127)
    return torch.where(zero_mask, torch.zeros_like(exp), exp)


# --------------------------------------------------------------------------- #
# Vectorized LUT quantization via searchsorted on the positive half of the
# LUT, replicating the baseline's argmin tie-breaking (lowest index wins,
# i.e. round-half-toward-zero) without ever materializing a (..., 256)
# distance tensor.
# --------------------------------------------------------------------------- #
def _vectorized_lut_quantize(
    scaled: torch.Tensor, format_name: str, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Quantize ``scaled`` to FP8 with the same semantics as the baseline
    ``MXFP8MatrixQuantizer._quantize_to_fp8`` (full-LUT ``argmin`` with
    PyTorch's lowest-index tie-breaking).

    Algorithm:
      1. Take the positive half of the LUT, ``lut_pos`` (sorted ascending).
      2. ``searchsorted`` gives the upper neighbour ``upper_idx`` such that
         ``lut_pos[upper_idx-1] <= |x| <= lut_pos[upper_idx]``.
      3. Pick the closer of the two neighbours; on a tie, pick the *lower*
         neighbour (smaller magnitude). This reproduces full-LUT argmin
         because all negative LUT entries are strictly farther from a
         positive ``x`` than the same-magnitude positive entry, and vice
         versa, and at a positive↔negative tie at exactly zero the lowest
         positive index wins.
      4. Re-apply the sign of ``x``. ``sign(0)*0 = +0``, which matches
         baseline (full argmin returns idx 0 = +0 for input 0).
    """
    lut_pos = _get_lut_pos(format_name)  # (128,)
    last_idx = lut_pos.numel() - 1  # 127

    sign = torch.sign(scaled)
    mag = scaled.abs()

    upper_idx = torch.searchsorted(lut_pos, mag).clamp(max=last_idx)
    lower_idx = (upper_idx - 1).clamp(min=0)

    upper_val = lut_pos[upper_idx]
    lower_val = lut_pos[lower_idx]

    # On equal distance, baseline argmin returns the lower (smaller-magnitude)
    # index, so we use ``<=`` to break ties toward the lower neighbour.
    pick_lower = (mag - lower_val) <= (upper_val - mag)
    chosen_mag = torch.where(pick_lower, lower_val, upper_val)

    snapped_fp32 = sign * chosen_mag

    # Canonicalize -0 to +0. Baseline-LUT argmin over the full 256-entry LUT
    # has +0 at idx 0 AND -0 at idx 128 with identical absolute distances; on
    # any tie at distance |x| (which always happens when chosen_mag == 0),
    # ``argmin`` returns the lower index → +0. ``sign(-x) * 0`` here would
    # otherwise produce -0 (different byte: 0x80 vs 0x00 in float8_e4m3fn).
    zero_mask = chosen_mag == 0
    snapped_fp32 = torch.where(zero_mask, torch.zeros_like(snapped_fp32), snapped_fp32)

    # Cast is lossless: every value in `snapped_fp32` is an exact FP8 grid
    # point, so RNE picks itself.
    return snapped_fp32.to(fp8_dtype)


# --------------------------------------------------------------------------- #
# Per-axis MX quantization. ``_quantize_axis_last`` matches axis=1 of the
# baseline (block-along-columns); ``_quantize_axis_first`` matches axis=0
# (block-along-rows) by transposing.
# --------------------------------------------------------------------------- #
def _quantize_axis_last(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize along the last axis. Returns ``(quant_fp8 (M, N),
    scale (M, padded_blocks), dequant_fp32 (M, N))``.

    ``padded_blocks`` is rounded up to an even number to match the baseline
    ``scale_matrix = torch.ones(((num_blocks + 1) // 2 * 2, ...))`` shape.
    """
    M, N = matrix.shape
    fmt = _FP8_FORMATS[format_name]
    fp8_dtype = fmt["torch_dtype"]
    fp8_emax = fmt["emax"]
    fp8_max = fmt["max_value"]

    num_blocks = (N + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    if padded_n != N:
        padded = torch.zeros(M, padded_n, dtype=matrix.dtype)
        padded[:, :N] = matrix
    else:
        padded = matrix

    blocks = padded.view(M, num_blocks, block_size)

    max_abs = blocks.abs().amax(dim=-1)  # (M, NB)
    exp = _e8m0_exp(max_abs, fp8_emax)  # (M, NB)
    scale = torch.exp2(exp.to(torch.float32))  # (M, NB)

    scaled = blocks / scale.unsqueeze(-1)  # (M, NB, BS)
    scaled_clamped = scaled.clamp(-fp8_max, fp8_max)

    if _QUANT_BACKEND == "native":
        quant_fp8 = scaled_clamped.to(fp8_dtype)
    else:
        quant_fp8 = _vectorized_lut_quantize(scaled_clamped, format_name, fp8_dtype)

    # Dequantize: cast back to fp32 (lossless, exact LUT value), multiply by
    # the (power-of-two) scale → exact reconstruction matching the baseline
    # `_dequantize_by_*` code path.
    dequant = quant_fp8.to(torch.float32) * scale.unsqueeze(-1)

    # Trim padding (no-op for K%32==0 cases).
    if padded_n != N:
        quant_fp8 = quant_fp8.reshape(M, padded_n)[:, :N].contiguous()
        dequant = dequant.reshape(M, padded_n)[:, :N].contiguous()
    else:
        quant_fp8 = quant_fp8.reshape(M, N)
        dequant = dequant.reshape(M, N)

    # Pad scale matrix to even num_blocks (matches baseline shape).
    padded_blocks = ((num_blocks + 1) // 2) * 2
    if padded_blocks != num_blocks:
        scale_padded = torch.ones((M, padded_blocks), dtype=torch.float32)
        scale_padded[:, :num_blocks] = scale
        scale = scale_padded

    return quant_fp8, scale, dequant


def _quantize_axis_first(
    matrix: torch.Tensor, format_name: str, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize along the first axis (axis=0 in baseline)."""
    qt, st, dt = _quantize_axis_last(matrix.t().contiguous(), format_name, block_size)
    return (qt.t().contiguous(), st.t().contiguous(), dt.t().contiguous())


def _quantize(
    matrix: torch.Tensor, format_name: str, axis: int, block_size: int = _BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if axis == 0:
        return _quantize_axis_first(matrix, format_name, block_size)
    if axis == 1:
        return _quantize_axis_last(matrix, format_name, block_size)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def gen_data_fp8_e4m3(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32) * 10.0 - 5.0
    quant_fp8, scale_fp32, dequant_fp32 = _quantize(matrix, "E4M3", axis)
    return (
        quant_fp8.to(torch.float8_e4m3fn),
        scale_fp32.to(torch.float8_e8m0fnu),
        dequant_fp32,
    )


def gen_data_fp8_e5m2(row, col, axis):
    matrix = torch.randn((row, col), dtype=torch.float32)
    quant_fp8, scale_fp32, dequant_fp32 = _quantize(matrix, "E5M2", axis)
    return (
        quant_fp8.to(torch.float8_e5m2),
        scale_fp32.to(torch.float8_e8m0fnu),
        dequant_fp32,
    )


def _resolve_workspace(data_root_cli: Optional[str]) -> str:
    """Parent directory for ``data/input`` and ``data/golden``."""
    if data_root_cli is not None:
        root = data_root_cli.strip()
        if root:
            return os.path.abspath(os.path.expanduser(root))
    return _SCRIPT_DIR


def gen_data(m, n, k, trans_a, trans_b, workspace: str) -> None:
    data_dir = os.path.join(workspace, "data")
    input_dir = os.path.join(data_dir, "input")
    golden_dir = os.path.join(data_dir, "golden")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)

    a_fp8, a_scale, a_fp32 = gen_data_fp8_e4m3(m, k, 1)
    b_fp8, b_scale, b_fp32 = gen_data_fp8_e4m3(k, n, 0)

    a_scale = a_scale.reshape(a_scale.shape[0], a_scale.shape[1] // 2, 2)
    b_scale = b_scale.reshape(b_scale.shape[0] // 2, 2, b_scale.shape[1])

    if trans_a == 1:
        a_fp8 = a_fp8.t()
        a_scale = a_scale.permute(1, 0, 2)

    if trans_b == 1:
        b_fp8 = b_fp8.t()
        b_scale = b_scale.permute(2, 0, 1)
    else:
        b_scale = b_scale.permute(0, 2, 1)

    a_np = torch.tensor(a_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    b_np = torch.tensor(b_fp8.flatten().untyped_storage(), dtype=torch.int8).numpy()
    a_np.tofile(os.path.join(input_dir, "a_8.bin"))
    b_np.tofile(os.path.join(input_dir, "b_8.bin"))

    a_scale_np = torch.tensor(
        a_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    b_scale_np = torch.tensor(
        b_scale.flatten().untyped_storage(), dtype=torch.int8
    ).numpy()
    a_scale_np.tofile(os.path.join(input_dir, "a_scale.bin"))
    b_scale_np.tofile(os.path.join(input_dir, "b_scale.bin"))

    c_fp32 = a_fp32 @ b_fp32
    c_np = c_fp32.numpy()
    c_np.tofile(os.path.join(golden_dir, "expected_data.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MX-FP8 inputs and FP32 golden under <data-root>/data/.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help="Directory under which data/input and data/golden are created. "
        "Default: this script's directory.",
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("trans_a", type=int)
    parser.add_argument("trans_b", type=int)
    args = parser.parse_args()
    workspace = _resolve_workspace(args.data_root)
    gen_data(args.m, args.n, args.k, args.trans_a, args.trans_b, workspace)
