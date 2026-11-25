/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_VEC_BINARY_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_VEC_BINARY_INTF_H

#include "kernel_tensor.h"
#include "ascendc_logger.h"

namespace AscendC {

template <typename T, bool isSetMask = true>
inline void Add(
    const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask,
    const uint8_t repeatTime, const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src0, src1, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Add, argsT, args);
}

template <typename T>
inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src0, src1, count};
    ASCENDC_LOG_CALL_T(Add, argsT, args);
}

template <typename T, bool isSetMask = true>
inline void Mul(
    const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, uint64_t mask,
    const uint8_t repeatTime, const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArgWithValue(isSetMask)};
    const std::vector<Arg> args = {dst, src0, src1, mask, repeatTime};
    ASCENDC_LOG_CALL_T(Mul, argsT, args);
}

template <typename T>
inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src0, src1, count};
    ASCENDC_LOG_CALL_T(Mul, argsT, args);
}

} // namespace AscendC

#endif