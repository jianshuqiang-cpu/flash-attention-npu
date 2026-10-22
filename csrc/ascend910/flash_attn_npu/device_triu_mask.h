/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <mutex>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Cache one 2048x2048 uint8 triu(1) mask for each NPU device used by this
// process. Each device is initialized lazily, so a rank does not allocate
// memory or create runtime state on unrelated visible devices. Initialization
// completes before the mask is published to other streams.
inline at::Tensor MakeDeviceTriuMask()
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    const auto ndev = c10_npu::device_count();
    TORCH_CHECK(ndev > 0, "No NPU device is available for the triangular mask cache");
    TORCH_CHECK(idx >= 0 && idx < ndev, "Invalid NPU device index ", idx, " for ", ndev, " visible devices");

    static const auto cached_device_count = ndev;
    static std::vector<std::once_flag> once_flags(static_cast<size_t>(cached_device_count));
    static std::vector<at::Tensor> masks(static_cast<size_t>(cached_device_count));
    TORCH_CHECK(ndev == cached_device_count, "The visible NPU device count changed after mask cache initialization");

    const auto cache_idx = static_cast<size_t>(idx);
    std::call_once(once_flags[cache_idx], [idx, cache_idx]() {
        constexpr int64_t dim = 2048;
        at::Tensor mask =
            at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
        mask.triu_(/*diagonal=*/1);
        masks[cache_idx] = std::move(mask);

        // call_once publishes host state, while this synchronization also
        // guarantees that callers on other streams cannot observe an in-flight
        // mask initialization.
        c10_npu::getCurrentNPUStream().synchronize();
    });

    return masks[cache_idx];
}

#endif
