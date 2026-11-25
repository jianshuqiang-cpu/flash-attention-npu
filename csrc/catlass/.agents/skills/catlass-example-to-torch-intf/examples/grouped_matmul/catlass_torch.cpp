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

at::Tensor catlass_torch::grouped_matmul(const at::Tensor& a, const at::Tensor& b, const at::Tensor& group_list)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2D tensors supported");
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "Tensor a must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "Tensor b must be on NPU");
    TORCH_CHECK(group_list.device().type() == c10::DeviceType::PrivateUse1, "group_list must be on NPU");
    TORCH_CHECK(a.dtype() == at::kHalf && b.dtype() == at::kHalf, "Only float16 supported");
    TORCH_CHECK(group_list.dtype() == at::kLong, "group_list must be int64");
    TORCH_CHECK(a.is_contiguous(), "Tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor b must be contiguous");
    TORCH_CHECK(group_list.is_contiguous(), "group_list must be contiguous");

    uint32_t M = static_cast<uint32_t>(a.size(0));
    uint32_t K = static_cast<uint32_t>(a.size(1));
    uint32_t N = static_cast<uint32_t>(b.size(1));
    uint32_t problemCount = static_cast<uint32_t>(group_list.size(0));

    auto c = at::empty({static_cast<int64_t>(M), static_cast<int64_t>(N)}, a.options());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    RUN_NPU_FUNC(
        catlass_kernel::grouped_matmul, coreNum, stream, a.data_ptr(), b.data_ptr(), c.data_ptr(),
        group_list.data_ptr(), problemCount, M, N, K);

    return c;
}

// In TORCH_LIBRARY block:
//   m.def("grouped_matmul(Tensor a, Tensor b, Tensor group_list) -> Tensor");

// In TORCH_LIBRARY_IMPL block:
//   m.impl("grouped_matmul", catlass_torch::grouped_matmul);
