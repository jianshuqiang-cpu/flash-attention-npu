/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// === Append to src/catlass_torch.cpp ===
// Requires these includes in the main file:
//   #include <torch/extension.h>
//   #include <torch_npu/csrc/core/npu/NPUStream.h>
//   #include <torch_npu/csrc/core/npu/NPUFormat.h>
//   #include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
//   #include <torch_npu/csrc/framework/utils/OpAdapter.h>
//   #include <tiling/platform/platform_ascendc.h>
//   #include "catlass_kernel.h"

at::Tensor catlass_torch::basic_matmul(const at::Tensor& a, const at::Tensor& b)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(a.size(1) == b.size(0), "Shape mismatch: a.size(1) != b.size(0)");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "Tensor a must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "Tensor b must be on NPU");
    TORCH_CHECK(a.dtype() == at::kHalf && b.dtype() == at::kHalf, "Only float16 supported");

    uint32_t M = static_cast<uint32_t>(a.size(0));
    uint32_t K = static_cast<uint32_t>(a.size(1));
    uint32_t N = static_cast<uint32_t>(b.size(1));

    auto c = at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)}, a.options());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    RUN_NPU_FUNC(catlass_kernel::basic_matmul, coreNum, stream, a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K);

    return c;
}

// In TORCH_LIBRARY block:
//   m.def("basic_matmul(Tensor a, Tensor b) -> Tensor");

// In TORCH_LIBRARY_IMPL block:
//   m.impl("basic_matmul", catlass_torch::basic_matmul);
