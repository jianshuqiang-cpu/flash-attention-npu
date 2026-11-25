# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import re

import pytest
import torch
import torch_npu
import torch_catlass
import math

from common import only_on_3510


GROUP_SIZES = (128, 128, 128, 128)
M_TOTAL = sum(GROUP_SIZES)
N = 256
K = 256

def _prefix_sum_group_list(group_sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(group_sizes, dtype=torch.int64).cumsum(0).npu()

def _grouped_matmul_reference(
    a: torch.Tensor, b: torch.Tensor, group_sizes: tuple[int, ...]
) -> torch.Tensor:
    expected = []
    offset = 0
    for group_id, group_size in enumerate(group_sizes):
        end = offset + group_size
        expected.append(a[offset:end].float() @ b[group_id].float())
        offset = end
    return torch.cat(expected, dim=0)

def group_matmul_reference(tensor_a, tensor_b, m_cumsum_list):
    if m_cumsum_list is None:
        return None
    group_num = len(m_cumsum_list)
    m = m_cumsum_list[-1]
    n = tensor_b.shape[-1]
    ab = torch.zeros(m, n, device="npu", dtype=torch.float32).to(torch.float32)

    for i in range(group_num):
        m_start = 0 if i == 0 else m_cumsum_list[i - 1]
        m_end = m_cumsum_list[i]
        a_in_group = tensor_a[m_start:m_end]
        b_in_group = tensor_b[i]
        ab_in_group = torch.matmul(a_in_group, b_in_group)
        ab[m_start:m_end, :] = ab_in_group
    return ab

def gelu_reference_origin(ab):
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * ab * (1.0 + torch.tanh(sqrt_2_over_pi * (ab + 0.044715 * torch.pow(ab, 3))))

def gelu_reference_sigmod(x: torch.Tensor) -> torch.Tensor:
    # 近似公式实现
    term = x + 0.044715 * torch.pow(x, 3)
    exponent = -1.595769 * term
    denominator = 1 + torch.exp(exponent)
    return x / denominator

def _grouped_matmul_slice_m_gelu_reference(tensor_a, tensor_b, m_cumsum_list, gelu_flag=0):
    ab = group_matmul_reference(tensor_a, tensor_b, m_cumsum_list)
    if gelu_flag == 0:
        return ab
    # gelu处理：torch.nn.functional.gelu(ab)
    gelu_out = gelu_reference_sigmod(ab)
    return ab, gelu_out

@only_on_3510
def test_grouped_matmul_slice_gelu():
    a = torch.randn(M_TOTAL, K, dtype=torch.float16)
    b = torch.randn(len(GROUP_SIZES), K, N, dtype=torch.float16)

    grouplist = _prefix_sum_group_list(GROUP_SIZES)
    grouplist_cpu = grouplist.cpu().to(torch.int64)

    a_npu = a.npu()
    b_npu = b.npu()

    result = torch_catlass.grouped_matmul_slice_m_gelu(a_npu, b_npu, grouplist)
    _, expected = _grouped_matmul_slice_m_gelu_reference(a, b, grouplist_cpu, gelu_flag=1)

    assert result.shape == (M_TOTAL, N)
    assert result.dtype == torch.float16
    assert result.device.type == "npu"
    assert torch.allclose(result.cpu().float(), expected.cpu(), rtol=1e-2, atol=1e-2), (
        f"max diff = {(result.cpu().float() - expected.cpu()).abs().max().item()}"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
