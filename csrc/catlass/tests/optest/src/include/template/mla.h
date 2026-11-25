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

#ifndef OPTEST_MLA_H
#define OPTEST_MLA_H

#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct MlaHost {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& query_nope,
        const at::Tensor& query_rope,
        const at::Tensor& key_cache,
        const at::Tensor& key_rope_cache,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& block_table,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t sparse_mode,
        CatlassKernel::MlaParams& params)
    {
        aclDataType queryDtype = TorchDtypeToAclDtype(query_nope.scalar_type());
        aclDataType keyDtype = TorchDtypeToAclDtype(key_cache.scalar_type());
        TORCH_CHECK(
            queryDtype == keyDtype,
            "query and key cache must have the same dtype");
        TORCH_CHECK(
            queryDtype == ACL_FLOAT16 || queryDtype == ACL_BF16,
            "mla supports float16 and bfloat16 only");

        TORCH_CHECK(
            actual_seq_lengths.scalar_type() == at::kInt || actual_seq_lengths.scalar_type() == at::kLong,
            "actual_seq_lengths must be int32 or int64");
        TORCH_CHECK(
            actual_seq_lengths_kv.scalar_type() == at::kInt || actual_seq_lengths_kv.scalar_type() == at::kLong,
            "actual_seq_lengths_kv must be int32 or int64");

        int64_t qNtokens = query_nope.size(0);
        int64_t embeddingSize = query_nope.size(2);
        int64_t qRopeHeadDim = query_rope.size(2);
        int64_t kvRopeHeadDim = key_rope_cache.size(3);
        int64_t batch = actual_seq_lengths.numel();
        TORCH_CHECK(
            batch == actual_seq_lengths_kv.numel(),
            "actual_seq_lengths and actual_seq_lengths_kv must have the same size");
        TORCH_CHECK(
            query_nope.size(1) == num_heads,
            "query_nope num_heads mismatch");
        TORCH_CHECK(
            query_rope.size(0) == qNtokens && query_rope.size(1) == num_heads,
            "query_rope shape mismatch");
        TORCH_CHECK(
            key_cache.size(2) == num_key_value_heads,
            "key_cache kv_heads mismatch");

        int64_t qSeqlen = actual_seq_lengths.max().item<int64_t>();
        int64_t kvSeqlen = actual_seq_lengths_kv.max().item<int64_t>();

        uint32_t maskType = 0;
        if (sparse_mode == 0) {
            maskType = 0;
        } else if (sparse_mode == 1) {
            maskType = 1;
        } else {
            throw std::runtime_error("sparse_mode of mla should be 0 or 1");
        }

        params.inputAddr.resize(5);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(query_nope.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(query_rope.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(key_cache.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(key_rope_cache.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(block_table.storage().data()));

        auto qSeqCpu = actual_seq_lengths.contiguous().cpu().to(at::kLong);
        auto kvSeqCpu = actual_seq_lengths_kv.contiguous().cpu().to(at::kLong);
        params.qSeqHost.resize(batch);
        params.kvSeqHost.resize(batch);
        for (int64_t i = 0; i < batch; ++i) {
            params.qSeqHost[static_cast<size_t>(i)] = static_cast<int32_t>(qSeqCpu[i].item<int64_t>());
            params.kvSeqHost[static_cast<size_t>(i)] = static_cast<int32_t>(kvSeqCpu[i].item<int64_t>());
        }

        params.qNtokens = static_cast<uint32_t>(qNtokens);
        params.batch = static_cast<uint32_t>(batch);
        params.qSeqlen = static_cast<uint32_t>(qSeqlen);
        params.kvSeqlen = static_cast<uint32_t>(kvSeqlen);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_key_value_heads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.qRopeHeadDim = static_cast<uint32_t>(qRopeHeadDim);
        params.kvRopeHeadDim = static_cast<uint32_t>(kvRopeHeadDim);
        params.numBlocks = static_cast<uint32_t>(key_cache.size(0));
        params.blockSize = static_cast<uint32_t>(key_cache.size(1));
        params.maskType = maskType;
        params.dataType = queryDtype;
    }

    static OutputType AllocOutput(CatlassKernel::MlaParams& params)
    {
        OutputType output = GetOutputTensor(
            {params.qNtokens, params.numHeads, params.embeddingSize},
            AclDtypeToTorchDtype(params.dataType));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }
};

struct MlaOp : MlaHost {
    static OutputType Run(
        const at::Tensor& query_nope,
        const at::Tensor& query_rope,
        const at::Tensor& key_cache,
        const at::Tensor& key_rope_cache,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const at::Tensor& block_table,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t sparse_mode)
    {
        CatlassKernel::MlaParams params;
        GetKernelInfo(
            query_nope, query_rope, key_cache, key_rope_cache,
            actual_seq_lengths, actual_seq_lengths_kv, block_table,
            num_heads, num_key_value_heads, sparse_mode, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::Mla, aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
