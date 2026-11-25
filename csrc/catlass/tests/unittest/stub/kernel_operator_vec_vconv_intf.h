/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_VEC_VCONV_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_VEC_VCONV_INTF_H

#include "kernel_tensor.h"
#include "ascendc_logger.h"

namespace AscendC {

enum class RoundMode
{
    CAST_NONE = 0,
    CAST_RINT = 1,
    CAST_FLOOR = 2,
    CAST_CEIL = 3,
    CAST_TRUNC = 4,
    CAST_POS_INF = 5,
    CAST_NEG_INF = 6
};

enum class SatMode
{
    SAT_NONE = 0,
    SAT_TRUNC = 1,
    SAT_RDN = 2,
    SAT_RUP = 3
};

template <typename T, typename U, bool isSetMask = true>
inline void Cast(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Cast, argsT, args);
}

template <typename T, typename U>
inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Cast, argsT, args);
}

template <typename T, typename U>
inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, RoundMode roundMode, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>()};
    const std::vector<Arg> args = {dst, src, roundMode, count};
    ASCENDC_LOG_CALL_T(Cast, argsT, args);
}

template <typename T, typename U, bool isSetMask = true>
inline void CastDequant(
    const LocalTensor<T>& dst, const LocalTensor<U>& src, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(CastDequant, argsT, args);
}

template <typename T, typename U>
inline void CastDequant(const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(CastDequant, argsT, args);
}

} // namespace AscendC

#endif