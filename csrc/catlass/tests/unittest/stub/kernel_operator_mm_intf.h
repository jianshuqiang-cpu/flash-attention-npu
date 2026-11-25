/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H

#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "ascendc_logger.h"

namespace AscendC {

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParamsV2& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);       
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParamsV2& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

#if (defined(__NPU_ARCH__) && __NPU_ARCH__==3510)
template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DMxParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);  
}
#endif // __NPU_ARCH__ == 3510

// Load 3D模式
template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData3DParamsV2<T>& loadDataParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<IsResetLoad3dConfig>()};
    const std::vector<Arg> args = {dst, src, loadDataParams};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData3DParamsV2Pro& loadDataParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, loadDataParams};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

// 辅助函数
inline void Load3DSetFMatrixCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    const std::vector<Arg> args = {l1H, l1W, padList};
    ASCENDC_LOG_CALL_T(Load3DSetFMatrixCal, {}, args);
}

inline void Load3DSetFMatrixCal(uint64_t regFMatrix)
{
    const std::vector<Arg> args = {regFMatrix};
    ASCENDC_LOG_CALL_T(Load3DSetFMatrixCal, {}, args);
}

// 带转置
template <typename T>
inline void LoadDataWithTranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2dTransposeParams& loadDataParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, loadDataParams};
    ASCENDC_LOG_CALL_T(LoadDataWithTranspose, argsT, args);
}

template <typename T>
inline void LoadDataWithTranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2dTransposeParamsV2& loadDataParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, loadDataParams};
    ASCENDC_LOG_CALL_T(LoadDataWithTranspose, argsT, args);
}

// Sparse version
template <typename T>
inline void LoadDataWithSparse(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint8_t> index, const LoadData2dParams& loadDataParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, index, loadDataParams};
    ASCENDC_LOG_CALL_T(LoadDataWithSparse, argsT, args);
}

template <typename T, typename U, typename S>
inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArg<S>()};
    const std::vector<Arg> args = {dst, fm, filter, params};
    ASCENDC_LOG_CALL_T(Mmad, argsT, args);
}

template <typename T, typename U, typename S, typename V>
inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias,
    const MmadParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArg<S>(), Arg::MakeArg<V>()};
    const std::vector<Arg> args = {dst, fm, filter, bias, params};
    ASCENDC_LOG_CALL_T(Mmad, argsT, args);
}

} // namespace AscendC

inline void load_cbuf_to_cb_mx(uint64_t dst, void* src, uint16_t x_start_pos, uint16_t y_start_pos, uint8_t x_step, uint8_t y_step, uint16_t src_stride, uint16_t dst_stride)
{
    const std::vector<Arg> args = {dst, src, x_start_pos, y_start_pos, x_step, y_step, src_stride, dst_stride};
    ASCENDC_LOG_CALL_T(load_cbuf_to_cb_mx, {}, args);
}

inline void load_cbuf_to_ca_mx(uint64_t dst, void* src, uint16_t x_start_pos, uint16_t y_start_pos, uint8_t x_step, uint8_t y_step, uint16_t src_stride, uint16_t dst_stride)
{
    const std::vector<Arg> args = {dst, src, x_start_pos, y_start_pos, x_step, y_step, src_stride, dst_stride};
    ASCENDC_LOG_CALL_T(load_cbuf_to_ca_mx, {}, args);
}

#endif // ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H
