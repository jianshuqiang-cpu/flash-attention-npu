#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch_npu
import math

import os
import sys
import random

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

M_LIST_POLICY_AVERAGE = 0
M_LIST_POLICY_RANDOM = 1
M_LIST_POLICY_RANDOM_FIRST_ZERO = 2
M_LIST_POLICY = M_LIST_POLICY_RANDOM


def is_similar(result, expected, rtol=0.01, atol=0.01):
    if result.shape != expected.shape:
        raise RuntimeError(
            "is_similar: shape not equal, result shape: {}, expected shape: {}".format(
                result.shape, expected.shape
            )
        )
    return torch.allclose(result, expected, rtol, atol)


def set_seed(seedValue):
    np.random.seed(seedValue)
    random.seed(seedValue)
    torch.manual_seed(seedValue)


def group_matmul_golden(tensor_a, tensor_b, m_cumsum_list):
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
        b_in_group = b_in_group
        ab_in_group = torch.matmul(a_in_group, b_in_group)
        ab[m_start:m_end, :] = ab_in_group
    return ab


def gelu_golden(ab):
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return (
        0.5
        * ab
        * (1.0 + torch.tanh(sqrt_2_over_pi * (ab + 0.044715 * torch.pow(ab, 3))))
    )


def gelu_golden_sigmod(x: torch.Tensor) -> torch.Tensor:
    # 近似公式实现
    term = x + 0.044715 * torch.pow(x, 3)
    exponent = -1.595769 * term
    denominator = 1 + torch.exp(exponent)
    return x / denominator


def gmm_gelu_golden(tensor_a, tensor_b, m_cumsum_list, gelu_flag=0):
    ab = group_matmul_golden(tensor_a, tensor_b, m_cumsum_list)
    if gelu_flag == 0:
        return ab
    # gelu处理：torch.nn.functional.gelu(ab)
    gelu_out = gelu_golden_sigmod(ab)
    return ab, gelu_out


def generate_m_list(m, group_num):
    #: m_list: m0 m0 m0 ... m1 m1 m1; m0 = m1 + 1 or m0数为0 ~ group_num-1
    m_list = torch.zeros(group_num).to(torch.int32)
    m_1 = m // group_num
    m_0_num = m - m_1 * group_num
    m_list[:m_0_num] = m_1 + 1
    m_list[m_0_num:] = m_1

    print(f"m_list={m_list}")

    m_cumsum_list = torch.cumsum(m_list, dim=0, dtype=torch.int64)
    return m_cumsum_list


def generate_m_list_random(m, group_num):
    # 随机生成 group_num-1 个分割点 [0, m]
    split = torch.randint(0, m + 1, size=(group_num,))
    split[group_num - 1] = m
    m_cumsum_list, _ = torch.sort(split)
    return m_cumsum_list


def generate_m_list_random_zero(m, group_num):
    # 随机生成 group_num-1 个分割点 [0, m]
    split = torch.randint(0, m + 1, size=(group_num,))
    split[group_num - 1] = m
    m_cumsum_list, _ = torch.sort(split)
    m_cumsum_list[0] = 0
    return m_cumsum_list


def run_test(group_num, m, n, k):
    print(f"group_num={group_num}, m={m}, n={n}, k={k}")

    tensor_a = (torch.rand(m, k, device="npu", dtype=torch.float16) - 0.5).to(
        torch.float16
    )
    tensor_b = (
        torch.rand(group_num, k, n, device="npu", dtype=torch.float16) - 0.5
    ).to(torch.float16)

    if M_LIST_POLICY == M_LIST_POLICY_RANDOM:
        m_cumsum_list = generate_m_list_random(m, group_num)
    elif M_LIST_POLICY == M_LIST_POLICY_RANDOM_FIRST_ZERO:
        m_cumsum_list = generate_m_list_random_zero(m, group_num)
    else:
        m_cumsum_list = generate_m_list(m, group_num)

    print(f"m_cumsum_list={m_cumsum_list}")

    tensor_a.cpu().numpy().astype(np.float16).tofile(
        os.path.join(WORKSPACE, "data", "tensor_a.bin")
    )
    tensor_b.cpu().numpy().astype(np.float16).tofile(
        os.path.join(WORKSPACE, "data", "tensor_b.bin")
    )

    mm_out, gelu_out = gmm_gelu_golden(tensor_a, tensor_b, m_cumsum_list, gelu_flag=1)

    gelu_out.cpu().numpy().astype(np.float32).tofile(
        os.path.join(WORKSPACE, "data", "golden_gelu_out.bin")
    )
    mm_out.cpu().numpy().astype(np.float32).tofile(
        os.path.join(WORKSPACE, "data", "golden_mm_out.bin")
    )
    m_cumsum_list.cpu().numpy().astype(np.int64).tofile(
        os.path.join(WORKSPACE, "data", "group_list.bin")
    )


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    print("参数总数：", len(sys.argv) - 1)  # 减去脚本名本身
    print("所有参数：", sys.argv)

    group_num = int(sys.argv[1])
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    k = int(sys.argv[4])
    device_id = int(sys.argv[5])

    torch.npu.set_device(device_id)
    set_seed(1)

    run_test(group_num, m, n, k)
