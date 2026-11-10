/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTN_DEVICE_TRIU_MASK_H
#define FLASH_ATTN_DEVICE_TRIU_MASK_H

#include <unordered_map>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// Initialize only the device used by this caller; eager cache entries are ready to use.
// Per-thread device caches avoid touching other workers' devices.
template <int64_t Dim = 2048>
inline at::Tensor MakeDeviceTriuMask()
{
    const auto idx = c10_npu::getCurrentNPUStream().device_index();
    static thread_local std::unordered_map<c10::DeviceIndex, at::Tensor> masks;
    auto found = masks.find(idx);
    if (found != masks.end()) {
        return found->second;
    }
    c10_npu::NPUGuard guard(idx);
    constexpr int64_t dim = Dim;
    at::Tensor mask =
        at::ones({dim, dim}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kPrivateUse1, idx)));
    mask.triu_(/*diagonal=*/1);
    if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::None) {
        // A captured allocation belongs to this graph, not the eager cache.
        return mask;
    }
    c10_npu::getCurrentNPUStream().synchronize();
    return masks.emplace(idx, std::move(mask)).first->second;
}

#endif
