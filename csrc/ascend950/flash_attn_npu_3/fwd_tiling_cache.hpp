/** Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0. */
#pragma once
#include <cstring>
#include <string>
#include <unordered_map>
#include <ATen/ATen.h>
#include "tilingdata.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace flash_attn_npu_950_v3 {
// Called only after current lengths are validated and a fresh complete tiling
// is generated. Neither tensor addresses nor declared maximum lengths identify
// a schedule. Cached device buffers are immutable and private to their stream.
inline at::Tensor UploadFwdTiling(const FAInferTilingData& tiling, const at::Device& device)
{
    static thread_local std::unordered_map<std::string, at::Tensor> cache;
    const bool eager = c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None;
    std::string key;
    if (eager) {
        const auto stream = c10_npu::getCurrentNPUStream(device.index());
        const auto deviceId = device.index();
        const auto streamId = stream.id();
        key.assign(reinterpret_cast<const char*>(&deviceId), sizeof(deviceId));
        key.append(reinterpret_cast<const char*>(&streamId), sizeof(streamId));
        // The producer clears the entire structure, including padding.
        key.append(reinterpret_cast<const char*>(&tiling), sizeof(tiling));
        const auto found = cache.find(key);
        if (found != cache.end()) return found->second;
    }
    auto cpu = at::empty({static_cast<int64_t>(sizeof(tiling))},
                         at::device(c10::kCPU).dtype(at::kByte).pinned_memory(eager));
    std::memcpy(cpu.data_ptr(), &tiling, sizeof(tiling));
    auto result = cpu.to(device, at::kByte, eager);
    if (eager) {
        if (cache.size() >= 32U) cache.erase(cache.begin());
        cache.emplace(std::move(key), result);
    }
    return result;
}
} // namespace flash_attn_npu_950_v3
