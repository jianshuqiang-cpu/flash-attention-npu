# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
import torch_npu


from common import only_on_3510


M, N, K = 256, 512, 1024
RTOL, ATOL = 1e-3, 1e-3

IN_DTYPE = torch.float32
OUT_DTYPE = "float32"
TRANS_A = False
TRANS_B = False
FORMAT_A = False
FORMAT_B = False
NEGATIVE_SLOPE = 0.1


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype.lower(), None)
        if resolved is None:
            raise ValueError(f"{dtype} is not a data type of torch")
        return resolved
    return dtype


def _compare_like_catlass_golden(result: torch.Tensor, expected: torch.Tensor, compute_num: int) -> bool:
    """Align with examples/common/golden/compare_data.hpp CompareData."""
    rtol = 1.0 / 128 if compute_num >= 2048 else 1.0 / 256
    diff = (result - expected).abs()
    threshold = rtol * torch.maximum(torch.ones_like(expected), expected.abs())
    return not (diff > threshold).any().item()


def _reference_matmul(
    a: torch.Tensor, b: torch.Tensor, transA: bool, transB: bool, out_dtype: torch.dtype
) -> torch.Tensor:
    if transA:
        a = a.t()
    if transB:
        b = b.t()
    return torch.matmul(a, b).to(out_dtype)


def _make_inputs(
    in_dtype: str | torch.dtype = IN_DTYPE,
    transA: bool = TRANS_A,
    transB: bool = TRANS_B,
    device: str = "npu",
):
    in_dtype = _resolve_dtype(in_dtype)
    torch.manual_seed(42)
    a_shape = (K, M) if transA else (M, K)
    b_shape = (N, K) if transB else (K, N)
    a = torch.empty(a_shape, device=device, dtype=in_dtype).uniform_(-5, 5)
    b = torch.empty(b_shape, device=device, dtype=in_dtype).uniform_(-5, 5)
    return a, b


@pytest.mark.parametrize(
    "mode",
    ["add", "add_ub", "bias", "leaky_relu", "sigmoid", "silu", "tanh"],
)
@only_on_3510
def test_ascend950_matmul_evg(mode: str):
    import torch_catlass
    from torch_catlass.ops.ascend950_matmul_evg import EvgPostprocessMode

    out_dtype = _resolve_dtype(OUT_DTYPE)
    a, b = _make_inputs(IN_DTYPE, TRANS_A, TRANS_B)
    matmul_out = _reference_matmul(a, b, TRANS_A, TRANS_B, out_dtype)

    extra = None
    kwargs: dict = {}
    if mode in ("add", "add_ub"):
        extra = torch.empty(M, N, device="npu", dtype=out_dtype).uniform_(-5, 5)
        expected = matmul_out + extra
    elif mode == "bias":
        extra = torch.empty(N, device="npu", dtype=out_dtype).uniform_(-5, 5)
        expected = matmul_out + extra
    elif mode == "leaky_relu":
        kwargs["negative_slope"] = NEGATIVE_SLOPE
        expected = F.leaky_relu(matmul_out, negative_slope=NEGATIVE_SLOPE)
    elif mode == "sigmoid":
        expected = torch.sigmoid(matmul_out)
    elif mode == "silu":
        expected = F.silu(matmul_out)
    elif mode == "tanh":
        expected = torch.tanh(matmul_out)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    result = torch_catlass.ascend950_matmul_evg(
        a,
        b,
        mode=EvgPostprocessMode(mode),
        extra=extra,
        outDType=OUT_DTYPE,
        transA=TRANS_A,
        transB=TRANS_B,
        formatA=FORMAT_A,
        formatB=FORMAT_B,
        **kwargs,
    )
    assert result.shape == (M, N)
    assert result.dtype == out_dtype
    assert result.device.type == "npu"

    if mode == "tanh":
        assert _compare_like_catlass_golden(result.cpu(), expected.cpu(), K), (
            f"max diff = {(result.cpu() - expected.cpu()).abs().max().item()}"
        )
    else:
        assert torch.allclose(result.cpu(), expected.cpu(), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
