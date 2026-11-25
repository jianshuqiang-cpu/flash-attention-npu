/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Tile shape scaling utility — scales K tile dimension proportionally
 *        to element byte width relative to a baseline dtype.
 *
 * 使用方式：
 *   using L1 = TileShapeScaler<half,       float,  GemmShape<128, 256, 256>>::type;
 *   // half=baseline → KScale=1, L1 = GemmShape<128, 256, 256>
 *
 *   using L1 = TileShapeScaler<float,      half,   GemmShape<128, 256, 256>>::type;
 *   // float vs half → KScale=2, L1 = GemmShape<128, 256, 128>
 *
 *   using L1 = TileShapeScaler<ElementA,   half,   GemmShape<128, 256, 256>>::type;
 *   // ElementA 是根据 JIT 宏推导的实际 dtype
 */

#ifndef OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_H
#define OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_H

#include <type_traits>

namespace CatlassKernel {

/**
 * @brief Compute the K-dimension scale factor: sizeof(Actual) / sizeof(Baseline).
 *
 * KScale > 1  → 实际元素更宽，K tile 等比缩小以匹配 UB 预算。
 * 要求 sizeof(Actual) 能被 sizeof(Baseline) 整除，K 能被 KScale 整除。
 */
template <typename ActualDtype, typename BaselineDtype>
struct KScaleFactor {
    static constexpr uint32_t value = sizeof(ActualDtype) / sizeof(BaselineDtype);
    static_assert(sizeof(ActualDtype) % sizeof(BaselineDtype) == 0,
                  "sizeof(Actual) must be a multiple of sizeof(Baseline)");
};

/**
 * @brief 将 GemmShape<M, N, K> 的 K 除以 KScale，M、N 不变。
 */
template <typename Shape, uint32_t KScale>
struct ScaleKTile {
    static_assert(KScale > 0, "KScale must be positive");
    using type = Shape;  // 主模板由特化覆盖
};

template <uint32_t M, uint32_t N, uint32_t K, uint32_t KScale>
struct ScaleKTile<Catlass::GemmShape<M, N, K>, KScale> {
    static_assert(K % KScale == 0, "K must be divisible by KScale");
    using type = Catlass::GemmShape<M, N, K / KScale>;
};

/**
 * @brief 按元素宽度等比缩放 tile shape。
 *
 * @tparam ActualDtype   实际数据类型（如 JIT 宏派生的 ElementA）
 * @tparam BaselineDtype 基线类型（通常为 half，即 shape 设计基准）
 * @tparam BaseShape     基线 tile shape（如 GemmShape<128, 256, 256>）
 *
 * 输出：如果 ActualDtype 比 BaselineDtype 宽 N 倍，K tile 缩小为 1/N。
 */
template <typename ActualDtype, typename BaselineDtype, typename BaseShape>
struct TileShapeScaler {
    static constexpr uint32_t kScale = KScaleFactor<ActualDtype, BaselineDtype>::value;
    using type = typename ScaleKTile<BaseShape, kScale>::type;
};

}  // namespace CatlassKernel

#endif  // OPTEST_KERNELS_COMMON_TILE_SHAPE_SCALER_H
