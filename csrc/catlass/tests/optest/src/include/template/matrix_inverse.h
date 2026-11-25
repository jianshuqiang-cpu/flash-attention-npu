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

#ifndef OPTEST_MATRIX_INVERSE_H
#define OPTEST_MATRIX_INVERSE_H

#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct MatrixInverseOp {
    static at::Tensor Run(const at::Tensor& A)
    {
        TORCH_CHECK(A.dim() == 2, "matrix_inverse expects a 2-D tensor (N x N)");
        TORCH_CHECK(A.size(0) == A.size(1), "matrix_inverse expects a square matrix, got ", A.size(0), "x", A.size(1));

        aclDataType aDtype = TorchDtypeToAclDtype(A.scalar_type());
        TORCH_CHECK(aDtype == ACL_FLOAT, "matrix_inverse currently supports float only, got ", A.scalar_type());

        // Clone input: kernel operates in-place, so we work on a copy.
        at::Tensor output = A.clone();

        CatlassKernel::MatrixInverseParams params;
        params.N = static_cast<uint32_t>(output.size(0));
        params.dataType = aDtype;
        params.inputAddr.resize(1);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::MatrixInverse, aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
