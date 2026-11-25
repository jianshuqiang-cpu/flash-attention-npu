/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_H
#define ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_H

#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "ascendc_logger.h"

namespace AscendC {
// 连续搬运
template <typename T>
inline void DataCopy(const LocalTensor<T>&dst, const GlobalTensor<T>& src, const uint32_t count) 
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);
}

// 非连续/连续搬运
template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);
}

template <typename T>
inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);
}

template <typename T>
inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Nd2NzParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

template <typename T>
inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const Dn2NzParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

// data copy enhanced
template <typename T, typename U>
inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>()};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

template <typename T, typename U>
inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>()};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(DataCopy, argsT, args);  
}

} // namespace AscendC

#endif
