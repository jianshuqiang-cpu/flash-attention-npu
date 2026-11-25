# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import math
from typing import Sequence, Tuple

import pytest
import torch
import torch_npu
import torch_catlass
torch_catlass.clear_jit_cache()

from mx_golden import (
    prepare_fp8_mx_inputs,
    prepare_fp4_mx_inputs,
    _move_kernel_inputs_to_npu
)

from common import only_on_3510

_verbose = False

# MX-FP8 format params: quant_type -> (emax, fp8_max)
_FP8_MX_FORMAT = {
    torch.float8_e4m3fn: (8, 448.0),
    torch.float8_e5m2: (15, 57344.0),
}


def prepare_fp8_mx_inputs_gmm(
    group_sizes: Sequence[int],
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    quant_type: torch.dtype = torch.float8_e4m3fn,
    device: str = "npu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build grouped MX-FP8 Slice-M inputs by reusing ``prepare_fp8_mx_inputs`` per group.

    ``prepare_fp8_mx_inputs`` builds a single (non-grouped) MX matmul. For grouped
    Slice-M we call it once per group and assemble the kernel tensors:

    - A is sliced along M: each group contributes an ``(m_g, K)`` slice; the slices
      are concatenated into the full ``(sum(group_sizes), K)`` operand.
    - B is per-group and stacked to ``(G, K, N)`` (``trans_b=False``) or
      ``(G, N, K)`` (``trans_b=True``).
    - ``expected`` is the concatenation of each group's dequant matmul,
      shape ``(sum(group_sizes), N)``.

    ``quant_type`` selects the MX-FP8 format (``float8_e4m3fn`` or ``float8_e5m2``).
    """
    group_sizes = tuple(int(s) for s in group_sizes)
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    if any(s <= 0 for s in group_sizes):
        raise ValueError(f"group_sizes must be positive, got {group_sizes}")
    if quant_type not in _FP8_MX_FORMAT:
        raise ValueError(f"unsupported quant_type: {quant_type}, supported: {list(_FP8_MX_FORMAT)}")

    emax, fp8_max = _FP8_MX_FORMAT[quant_type]
    a_concat_dim = 1 if trans_a else 0

    a_list = []
    a_scale_list = []
    b_list = []
    b_scale_list = []
    expected_list = []
    for group_size in group_sizes:
        a_g, b_g, a_scale_g, b_scale_g, expected_g = prepare_fp8_mx_inputs(
            group_size, n, k,
            trans_a=trans_a, trans_b=trans_b, device="cpu",
            fp8_dtype=quant_type, emax=emax, fp8_max=fp8_max,
        )
        a_list.append(a_g)
        a_scale_list.append(a_scale_g)
        b_list.append(b_g)
        b_scale_list.append(b_scale_g)
        expected_list.append(expected_g)

    a_fp8 = torch.cat(a_list, dim=a_concat_dim).contiguous()
    a_scale = torch.cat(a_scale_list, dim=a_concat_dim).contiguous()
    b_fp8 = torch.stack(b_list, dim=0).contiguous()
    b_scale = torch.stack(b_scale_list, dim=0).contiguous()
    expected = torch.cat(expected_list, dim=0)

    if device == "npu":
        a_fp8 = a_fp8.npu()
        b_fp8 = b_fp8.npu()
        a_scale = a_scale.npu()
        b_scale = b_scale.npu()
    return a_fp8, b_fp8, a_scale, b_scale, expected

def prepare_fp4_mx_inputs_gmm(
    group_sizes: Sequence[int],
    n: int, 
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
    device: str = "npu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_sizes = tuple(int(s) for s in group_sizes)
    if not group_sizes:
        raise ValueError("group_sizes must not be empty")
    if any(s <= 0 for s in group_sizes):
        raise ValueError(f"group_sizes must be positive, got {group_sizes}")
    m = sum(group_sizes)

    a_packed_list = []
    a_scale_list = []
    b_packed_list = []
    b_scale_list = []
    expected_list = []

    for group_size in group_sizes:
        a_g, b_g, a_scale_g, b_scale_g, expected_g = prepare_fp4_mx_inputs(
            group_size, n, k, trans_a=trans_a, trans_b=trans_b, device="cpu",
        )
        # Extract tightly-packed FP4 front bytes from the over-allocated float4 buffers
        a_packed_len = group_size * math.ceil(k / 2)
        a_packed_list.append(a_g.view(torch.uint8).flatten()[:a_packed_len])
        b_packed_len = (
            k * math.ceil(n / 2) if not trans_b else n * math.ceil(k / 2)
        )
        b_packed_list.append(b_g.view(torch.uint8).flatten()[:b_packed_len])
        a_scale_list.append(a_scale_g)
        b_scale_list.append(b_scale_g)
        expected_list.append(expected_g)

    # Assemble A: contiguous packed rows → (M, K) logical shape
    a_packed_concat = torch.cat(a_packed_list)
    a = torch.empty((m, k), dtype=torch.float4_e2m1fn_x2)
    a.view(torch.uint8).flatten()[: a_packed_concat.numel()].copy_(a_packed_concat)
    a_scale = torch.cat(a_scale_list, dim=0).contiguous()

    # Assemble B: contiguous packed groups → (G, K, N) / (G, N, K) logical shape
    problem_count = len(group_sizes)
    b_packed_concat = torch.cat(b_packed_list)
    if trans_b:
        b = torch.empty((problem_count, n, k), dtype=torch.float4_e2m1fn_x2)
    else:
        b = torch.empty((problem_count, k, n), dtype=torch.float4_e2m1fn_x2)
    b.view(torch.uint8).flatten()[: b_packed_concat.numel()].copy_(b_packed_concat)
    b_scale = torch.stack(b_scale_list, dim=0).contiguous()

    expected = torch.cat(expected_list, dim=0)
    group_list = torch.tensor(group_sizes, dtype=torch.int64)

    if device == "npu":
        a, b, a_scale, b_scale = _move_kernel_inputs_to_npu(a, b, a_scale, b_scale)
        group_list = group_list.npu()
    
    return a, b, a_scale, b_scale, expected, group_list

@pytest.mark.parametrize("trans_b", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize("quant_type", [torch.float8_e4m3fn, torch.float8_e5m2,], ids=["e4m3fn", "e5m2"])
@pytest.mark.parametrize(
    "enable_aswt,enable_preload",
    [
        (False, False),
        (True, True),
    ],
    ids=["base", "aswt_preload"],
)
@only_on_3510
def test_ascend950_mx_grouped_matmul_slice_m_mxfp8(
    trans_b, quant_type, enable_aswt, enable_preload
):
    group_num, m, n, k = 2, 256, 512, 1024
    group_sizes = [m // group_num] * group_num
    group_list = torch.tensor(group_sizes, dtype=torch.int64)
    trans_a = False

    a_fp8, b_fp8, a_scale, b_scale, expected = prepare_fp8_mx_inputs_gmm(
        group_sizes, n, k, trans_a=trans_a, trans_b=trans_b,
        quant_type=quant_type, device="npu",
    )

    ret = torch_catlass.ascend950_mx_grouped_matmul_slice_m(
        a_fp8, b_fp8, group_list.npu(),
        a_scale, b_scale,
        trans_a, trans_b,
        enable_aswt=enable_aswt, enable_preload=enable_preload,
    )

    assert ret.shape == (m, n)
    assert ret.dtype == torch.float32
    assert ret.device.type == "npu"

    rtol, atol = 1e-2, 1e-2
    assert torch.allclose(ret.cpu(), expected, rtol=rtol, atol=atol), (
        f"trans_a={trans_a}, trans_b={trans_b}, quant_type={quant_type}, "
        f"enable_aswt={enable_aswt}, enable_preload={enable_preload}: max diff = "
        f"{(ret.cpu() - expected).abs().max().item()}"
    )

@only_on_3510
@pytest.mark.parametrize("trans_b", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize(
    "enable_aswt,enable_preload",
    [
        (False, False),
        (True, True),
    ],
    ids=["base", "aswt_preload"],
)
def test_ascend950_mx_grouped_matmul_slice_m_fp4(trans_b, enable_aswt, enable_preload):
    """FP4 grouped MX matmul Slice-M
    """
    group_num, m, n, k = 2, 4, 18, 220
    group_sizes = [m // group_num] * group_num
    trans_a = False

    a_npu, b_npu, a_scale_npu, b_scale_npu, expected, group_list_npu = prepare_fp4_mx_inputs_gmm(
        group_sizes, n, k, trans_a=trans_a, trans_b=trans_b, device="npu",
    )

    result = torch_catlass.ascend950_mx_grouped_matmul_slice_m(
        a_npu, b_npu, group_list_npu, 
        a_scale_npu, b_scale_npu, 
        trans_a, trans_b,
        enable_aswt=enable_aswt, enable_preload=enable_preload,
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.float32
    assert result.device.type == "npu"

    if _verbose:
        print(f"detailed tensor info: \nnpu result = {result.cpu().numpy().flatten()}\ncpu golden: {expected.numpy().flatten()}")

    rtol, atol = 1e-1, 1e-1
    assert torch.allclose(result.cpu(), expected, rtol=rtol, atol=atol), (
        f"trans_b={trans_b}: max diff = {(result.cpu() - expected).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
