/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_SYS_VAR_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_SYS_VAR_INTF_H

#include "ascendc_logger.h"

namespace AscendC {

inline int64_t GetBlockNum()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetBlockNum, args);
    return 1;
}

inline int64_t GetBlockIdx()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetBlockIdx, args);
    return 0;
}

inline int64_t GetSubBlockIdx()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetSubBlockIdx, args);
    return 0;
}

inline int64_t GetSubBlockNum()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetSubBlockNum, args);
    return 1;
}

inline int64_t GetTaskRatio()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetTaskRatio, args);
    return 1;
}

inline int16_t GetDataBlockSizeInBytes()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetDataBlockSizeInBytes, args);
    return 32;
}

inline void GetArchVersion(uint32_t& coreVersion)
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetArchVersion, args);
    coreVersion = 2201;
}

inline int64_t GetProgramCounter()
{

    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetProgramCounter, args);
    return 0;
}

inline void Trap()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(Trap, args);
}

inline int64_t GetSystemCycle()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetSystemCycle, args);
    return 0;
}

inline const  uint32_t GetUBSizeInBytes()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetUBSizeInBytes, args);
    return 256 * 1024;
}

inline uint32_t GetVecLen() 
{
    return 128;
}

inline uint32_t GetRuntimeUBSize()
{
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL(GetRuntimeUBSize, args);
    return 256 * 1024;
}

} // namespace AscendC

#endif