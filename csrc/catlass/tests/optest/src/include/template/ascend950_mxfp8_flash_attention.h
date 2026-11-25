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

#ifndef OPTEST_ASCEND950_MXFP8_FLASH_ATTENTION_H
#define OPTEST_ASCEND950_MXFP8_FLASH_ATTENTION_H

#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct Ascend950MxFp8FlashAttentionHost {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& atten_mask,
        const at::Tensor& block_table,
        const at::Tensor& q_scale,
        const at::Tensor& k_scale,
        const at::Tensor& v_scale,
        const c10::optional<at::Tensor>& p_scale,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t sparse_mode,
        CatlassKernel::Ascend950MxFp8FlashAttentionParams& params)
    {
        TORCH_CHECK(input_layout == "TND", "input_layout of ascend950_fp8_mx_flash_attention_infer only supports TND");

        aclDataType queryDtype = TorchDtypeToAclDtype(query.scalar_type());
        aclDataType keyDtype = TorchDtypeToAclDtype(key.scalar_type());
        aclDataType valueDtype = TorchDtypeToAclDtype(value.scalar_type());
        TORCH_CHECK(
            queryDtype == keyDtype && queryDtype == valueDtype,
            "query, key and value must have the same dtype");

        int64_t qNtokens = query.size(0);
        int64_t embeddingSize = query.size(2);
        int64_t batch = actual_seq_lengths.numel();
        TORCH_CHECK(
            batch == actual_seq_lengths_kv.numel(),
            "actual_seq_lengths and actual_seq_lengths_kv must have the same size");

        int64_t qSeqlen = actual_seq_lengths.max().item<int64_t>();
        int64_t kvSeqlen = actual_seq_lengths_kv.max().item<int64_t>();

        uint32_t maskType = 0;
        if (sparse_mode == 0) {
            maskType = 0;
        } else if (sparse_mode == 1) {
            maskType = 1;
        } else {
            throw std::runtime_error("sparse_mode of ascend950_fp8_mx_flash_attention_infer should be 0 or 1");
        }

        uint32_t usePscale = p_scale.has_value() ? 1 : 0;

        params.inputAddr.resize(usePscale ? 11 : 10);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths_kv.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(query.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(key.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(value.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(atten_mask.storage().data()));
        params.inputAddr[6] = static_cast<uint8_t*>(const_cast<void*>(block_table.storage().data()));
        params.inputAddr[7] = static_cast<uint8_t*>(const_cast<void*>(q_scale.storage().data()));
        params.inputAddr[8] = static_cast<uint8_t*>(const_cast<void*>(k_scale.storage().data()));
        params.inputAddr[9] = static_cast<uint8_t*>(const_cast<void*>(v_scale.storage().data()));
        if (usePscale) {
            params.inputAddr[10] = static_cast<uint8_t*>(const_cast<void*>(p_scale->storage().data()));
        }

        params.qNtokens = static_cast<uint32_t>(qNtokens);
        params.batch = static_cast<uint32_t>(batch);
        params.qSeqlen = static_cast<uint32_t>(qSeqlen);
        params.kvSeqlen = static_cast<uint32_t>(kvSeqlen);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_key_value_heads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.blockSize = 128;
        params.maskType = maskType;
        params.usePscale = usePscale;

        aclDataType outDtype = ACL_FLOAT16;
        if (queryDtype == ACL_FLOAT16) {
            outDtype = ACL_FLOAT16;
        } else if (queryDtype == ACL_BF16) {
            outDtype = ACL_BF16;
        }
        params.dataType = outDtype;
    }

    static OutputType AllocOutput(CatlassKernel::Ascend950MxFp8FlashAttentionParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.qNtokens, params.numHeads, params.embeddingSize},
            AclDtypeToTorchDtype(params.dataType));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }
};

struct Ascend950MxFp8FlashAttentionInferOp : Ascend950MxFp8FlashAttentionHost {
    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& atten_mask,
        const at::Tensor& block_table,
        const at::Tensor& q_scale,
        const at::Tensor& k_scale,
        const at::Tensor& v_scale,
        const c10::optional<at::Tensor>& p_scale,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t sparse_mode)
    {
        CatlassKernel::Ascend950MxFp8FlashAttentionParams params;
        GetKernelInfo(
            query, key, value, actual_seq_lengths, actual_seq_lengths_kv,
            atten_mask, block_table, q_scale, k_scale, v_scale, p_scale,
            input_layout, num_heads, num_key_value_heads, sparse_mode, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::Ascend950MxFp8FlashAttentionInfer, aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
