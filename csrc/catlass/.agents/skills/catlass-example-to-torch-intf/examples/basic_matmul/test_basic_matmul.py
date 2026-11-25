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

M, K, N = 1024, 512, 768

a = torch.randn(M, K, dtype=torch.float16, device="npu")
b = torch.randn(K, N, dtype=torch.float16, device="npu")

c = torch.ops.catlass.basic_matmul(a, b)
golden = torch.mm(a, b)

print(f"A: {a.shape}, B: {b.shape}, C: {c.shape}")
print(f"Max diff: {(c - golden).abs().max().item()}")
