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

#ifndef OPTEST_CONV_BIAS_H
#define OPTEST_CONV_BIAS_H

#include <stdexcept>
#include <vector>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct ConvBiasOp {
    using OutputType = at::Tensor;

    static OutputType Run(
        at::Tensor fmap, at::Tensor weight, at::Tensor bias, const std::vector<int64_t>& fmapRelated,
        const std::vector<int64_t>& filterRelated, const std::vector<int64_t>& strideList,
        const std::vector<int64_t>& padList, const std::vector<int64_t>& dilationList, int64_t outNumel)
    {
        CatlassKernel::ConvParams params;

        params.inputDataType = ACL_FLOAT16;
        params.biasDataType = ACL_FLOAT16;
        params.outputDataType = ACL_FLOAT16;

        params.fmapRelated.assign(fmapRelated.begin(), fmapRelated.end());
        params.filterRelated.assign(filterRelated.begin(), filterRelated.end());
        params.strideList.assign(strideList.begin(), strideList.end());
        params.padList.assign(padList.begin(), padList.end());
        params.dilationList.assign(dilationList.begin(), dilationList.end());

        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(fmap.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(weight.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(bias.storage().data()));

        auto output = GetOutputTensor({outNumel}, AclDtypeToTorchDtype(ACL_FLOAT16));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        CatlassKernel::ConvBias(aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
