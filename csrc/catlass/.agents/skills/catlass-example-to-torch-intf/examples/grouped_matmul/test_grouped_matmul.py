# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of
# the License.

import torch
import torch_npu
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, "build", "libcatlass.so")
torch.ops.load_library(lib_path)

test_cases = [
    {
        "groups": [(256, 512), (128, 512)],
        "K": 512,
        "N": 256,
    },
    {
        "groups": [(128, 1024), (256, 1024), (64, 1024)],
        "K": 1024,
        "N": 128,
    },
    {
        "groups": [(512, 256)],
        "K": 256,
        "N": 512,
    },
]

all_passed = True

for i, tc in enumerate(test_cases):
    groups = tc["groups"]
    K = tc["K"]
    N = tc["N"]

    a_list = []
    b_list = []
    golden_list = []
    group_list_vals = []
    cumulative_m = 0

    for m, k in groups:
        a_g = torch.randn(m, k, dtype=torch.float16, device="npu")
        b_g = torch.randn(k, N, dtype=torch.float16, device="npu")
        a_list.append(a_g)
        b_list.append(b_g)
        golden_list.append(torch.mm(a_g, b_g))
        cumulative_m += m
        group_list_vals.append(cumulative_m)

    a = torch.cat(a_list, dim=0)
    b = torch.cat(b_list, dim=0)
    group_list = torch.tensor(group_list_vals, dtype=torch.int64, device="npu")

    c = torch.ops.catlass.grouped_matmul(a, b, group_list)

    golden = torch.cat(golden_list, dim=0)

    max_diff = (c - golden).abs().max().item()
    mean_diff = (c - golden).abs().mean().item()
    passed = max_diff < 1.0

    group_desc = ", ".join([f"({m},{k})" for m, k in groups])
    print(
        f"Case {i}: groups=[{group_desc}], K={K}, N={N}: " f"max_diff={
            max_diff:.4f}, mean_diff={
            mean_diff:.4f} {
                'PASS' if passed else 'FAIL'}")
    if not passed:
        all_passed = False

print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
