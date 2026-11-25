/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_VEC_UNARY_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_VEC_UNARY_INTF_H

#include "kernel_tensor.h"
#include "ascendc_logger.h"

namespace AscendC {

enum class MaskMode
{
    NORMAL,
    COUNTER,
    PRED
};

template <typename T>
inline void SetMaskCount()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL_T(SetMaskCount, {}, args);
}

template <typename T, MaskMode mode>
inline void SetVectorMask(int32_t count)
{
    const std::vector<Arg> args = {count};
    ASCENDC_LOG_CALL_T(SetVectorMask, {}, args);
}

inline void SetMaskNorm()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL_T(SetMaskNorm, {}, args);
}

inline void ResetMask()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL_T(ResetMask, {}, args);
}

template <typename T, bool isSetMask = true>
inline void Muls(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, T scalar, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, scalar, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Muls, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, T scalar, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, scalar, count};
    ASCENDC_LOG_CALL_T(Muls, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Adds(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, T scalar, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, scalar, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Adds, argsT, args);
}

template <typename T>
inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, T scalar, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, scalar, count};
    ASCENDC_LOG_CALL_T(Adds, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Relu(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[], const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Relu, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Relu(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Relu, argsT, args);
}

template <typename T>
inline void Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Relu, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Exp(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Exp, argsT, args);
}

template <typename T>
inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Exp, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Tanh(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTime,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Tanh, argsT, args);
}

template <typename T>
inline void Tanh(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Tanh, argsT, args);
}

} // namespace AscendC

#endif