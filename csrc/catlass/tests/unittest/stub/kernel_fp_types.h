/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_FP_TYPES_H
#define ASCENDC_STUB_KERNEL_FP_TYPES_H

#include <cstdint>
#include <type_traits>

// FP8/FP4 type stubs shared between kernel_operator.h and kernel_struct_mm.h
// to break the circular include dependency without pulling in the full operator.

namespace AscendC {

struct fp8_e8m0_t { uint8_t _; };
struct fp8_e5m2_t { uint8_t _; };
struct fp8_e4m3fn_t { uint8_t _; };
struct hifloat8_t { uint8_t _; };
struct fp4x2_e1m2_t { uint8_t _; };
struct fp4x2_e2m1_t { uint8_t _; };
struct mx_fp8_e4m3_t { uint8_t _; };
struct mx_fp8_e5m2_t { uint8_t _; };

namespace Std {
template <typename T, typename... Types>
struct is_one_of : std::disjunction<std::is_same<T, Types>...> {};

template <typename T, typename... Types>
inline constexpr bool is_one_of_v = is_one_of<T, Types...>::value;
} // namespace Std

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_FP_TYPES_H
