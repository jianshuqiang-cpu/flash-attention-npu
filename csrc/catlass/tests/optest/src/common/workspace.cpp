/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/workspace.h"

#include <ATen/Tensor.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
namespace catlass_torch {
namespace common {

aclError workspaceMalloc(void** wkspAddr, size_t wkspSize)
{
    at::TensorOptions options = at::TensorOptions().dtype(torch::kInt8).device(torch_npu::utils::get_npu_device_type());
    at::Tensor workspaceTensor = at::empty({static_cast<int64_t>(wkspSize)}, options);
    *wkspAddr = const_cast<void*>(workspaceTensor.storage().data());
    return ACL_SUCCESS;
}

} // namespace common
} // namespace catlass_torch
