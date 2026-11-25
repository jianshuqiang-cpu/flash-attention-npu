/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_FIXPIPE_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_FIXPIPE_INTF_H

#include "kernel_tensor.h"
#include "kernel_struct_fixpipe.h"
#include "ascendc_logger.h"

namespace AscendC {

// Fixpipe configuration
template <typename T>
inline void SetFixPipeConfig(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false) {
        const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
        const std::vector<Arg> args = {reluPre, quantPre, isUnitFlag};
        ASCENDC_LOG_CALL_T(SetFixPipeConfig, argsT, args);
}

template <typename T, bool setRelu = false>
inline void SetFixPipeConfig(const LocalTensor<T> &preData, bool isUnitFlag = false) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {preData, isUnitFlag};
    ASCENDC_LOG_CALL_T(SetFixPipeConfig, argsT, args);
}

// Fixpipe addr configuration
template <typename T>
inline void SetFixPipeAddr(const LocalTensor<T> &eleWiseData, uint16_t c0ChStride) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {eleWiseData, c0ChStride};
    ASCENDC_LOG_CALL_T(SetFixPipeAddr, argsT, args);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint32_t dstNdStride) {
    const std::vector<Arg> args = {ndNum, srcNdStride, dstNdStride};
    ASCENDC_LOG_CALL_T(SetFixpipeNz2ndFlag, {}, args);
}
#else // __NPU_ARCH__ == 3510
inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride) {
    const std::vector<Arg> args = {ndNum, srcNdStride, dstNdStride};
    ASCENDC_LOG_CALL_T(SetFixpipeNz2ndFlag, {}, args);
}
#endif

// Pre configuration
inline void SetFixpipePreQuantFlag(uint64_t config) {
    const std::vector<Arg> args = {config};
    ASCENDC_LOG_CALL_T(SetFixpipePreQuantFlag, {}, args);
}

// Clip-relu configuration
inline void SetFixPipeClipRelu(uint64_t config) {
    const std::vector<Arg> args = {config};
    ASCENDC_LOG_CALL_T(SetFixPipeClipRelu, {}, args);
}

// Fixpipe (L0C->L1)
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    const FixpipeParamsV220& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, 
    const FixpipeParamsV220& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

// L0C->GM deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
     const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsV220& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, cbufWorkspace, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// L0C->L1
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    const FixpipeParamsArch3510<config.format>& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsArch3510<config.format>& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, cbufWorkspace, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, 
    const FixpipeParamsArch3510<config.format>& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}

template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, 
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsArch3510<config.format>& intriParams) {
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArgWithValue<FixpipeConfig>(config)};
    const std::vector<Arg> args = {dst, src, cbufWorkspace, intriParams};
    ASCENDC_LOG_CALL_T(Fixpipe, argsT, args);
}
#endif // __NPU_ARCH__ == 3510


} // namespace AscendC

#endif