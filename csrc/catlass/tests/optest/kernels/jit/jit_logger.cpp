/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "jit_logger.h"

#include <cstdlib>

#include "jit_config.h"

extern "C" int GetJitLogLevel()
{
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv(CatlassKernel::JitConfig::kLogLevelEnv);
        level = env ? std::atoi(env) : 0;
        if (level < static_cast<int>(JitLogLevel::None))
            level = static_cast<int>(JitLogLevel::None);
        if (level > static_cast<int>(JitLogLevel::Debug))
            level = static_cast<int>(JitLogLevel::Debug);
    }
    return level;
}
