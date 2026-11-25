/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "type_utils.hpp"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

torch::Tensor GetOutputTensor(const std::vector<int64_t>& shape, const torch::Dtype dtype)
{
    at::TensorOptions options = at::TensorOptions();
    options =
        options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_ND);
}

torch::Dtype TypeStrToTorchDtype(const std::string& typeStr)
{
    return test_utils::TypeCast<std::string, torch::Dtype>(typeStr);
}

aclDataType TorchDtypeToAclDtype(const torch::Dtype torchDtype)
{
    return test_utils::TypeCast<torch::Dtype, aclDataType>(torchDtype);
}

torch::Dtype AclDtypeToTorchDtype(const aclDataType aclDtype)
{
    return test_utils::TypeCast<aclDataType, torch::Dtype>(aclDtype);
}

[[nodiscard]] TransposeStatus GetTransposeStatus(const at::Tensor& mat)
{
    constexpr int64_t kFormatND = 2;
    constexpr int64_t kFormatNZ = 29;

    int64_t npuFormat = at_npu::native::get_npu_format(mat);

    if (!mat.is_contiguous() && npuFormat == kFormatND) {
        return TransposeStatus::NON_CONTINUOUS;
    }

    if (mat.dim() < 2) {
        return TransposeStatus::NO_TRANSPOSE;
    }

    const auto& strides = mat.strides();
    const auto& shape = mat.sizes();

    int64_t dimA = shape[mat.dim() - 2];
    int64_t dimB = shape[mat.dim() - 1];
    int64_t strideA = strides[mat.dim() - 2];
    int64_t strideB = strides[mat.dim() - 1];

    if (npuFormat == kFormatND) {
        if (strideA == dimB && strideB == 1) {
            return TransposeStatus::NO_TRANSPOSE;
        }
        if (strideA == 1 && strideB == dimA) {
            return TransposeStatus::TRANSPOSE;
        }
        return TransposeStatus::NON_CONTINUOUS;
    }

    if (npuFormat == kFormatNZ) {
        if (strideA == dimB && strideB == 1) {
            return TransposeStatus::LAYOUT_ZN;
        }
        if (strideA == 1 && strideB == dimA) {
            return TransposeStatus::LAYOUT_NZ;
        }
        return TransposeStatus::NON_CONTINUOUS;
    }

    return TransposeStatus::NON_CONTINUOUS;
}

} // namespace CatlassKernelWrapper
