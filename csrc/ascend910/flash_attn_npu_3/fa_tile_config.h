/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#pragma once

#include <cstdint>

// Shared by host tiling, AICPU metadata and the AIC/AIV kernels.
namespace fa_tile_config {
constexpr uint32_t Q_ROWS = 256;
constexpr uint32_t APPEND_Q_ROWS = 128;
constexpr uint32_t MAX_KV_COLUMNS = 512;
constexpr uint64_t WORKSPACE_ELEMENTS = static_cast<uint64_t>(Q_ROWS) * MAX_KV_COLUMNS;
} // namespace fa_tile_config
