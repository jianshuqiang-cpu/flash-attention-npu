/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */


#ifndef OPTEST_TORCH_UTILS_H
#define OPTEST_TORCH_UTILS_H

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace CatlassKernelWrapper {

enum class TransposeStatus : uint32_t
{
    NO_TRANSPOSE = 0,
    TRANSPOSE = 1,
    NON_CONTINUOUS = 2,
    LAYOUT_ZN = 3,
    LAYOUT_NZ = 4
};

/**
 * @brief Allocate an NPU output tensor with ND format.
 * @param shape Output tensor shape.
 * @param dtype Output tensor dtype.
 * @return Newly allocated tensor on the active NPU device.
 */
torch::Tensor GetOutputTensor(const std::vector<int64_t>& shape, torch::Dtype dtype);

/**
 * @brief Convert a canonical dtype string to a torch dtype.
 * @param typeStr Name such as "float16", "float", or "bfloat16".
 * @return Matching torch dtype.
 */
torch::Dtype TypeStrToTorchDtype(const std::string& typeStr);

/**
 * @brief Convert a torch dtype to the corresponding ACL dtype.
 * @param torchDtype PyTorch scalar dtype.
 * @return Matching ACL dtype.
 */
aclDataType TorchDtypeToAclDtype(torch::Dtype torchDtype);

/**
 * @brief Convert an ACL dtype to the corresponding torch dtype.
 * @param aclDtype ACL dtype value.
 * @return Matching PyTorch scalar dtype.
 */
torch::Dtype AclDtypeToTorchDtype(aclDataType aclDtype);

/**
 * @brief Classify the logical layout of a 2D tensor view.
 * @param mat Tensor whose last two dimensions are interpreted as a matrix.
 * @return Whether the matrix is normal, transposed, NZ-formatted, or unsupported.
 */
TransposeStatus GetTransposeStatus(const at::Tensor& mat);

} // namespace CatlassKernelWrapper

#endif // OPTEST_TORCH_UTILS_H
