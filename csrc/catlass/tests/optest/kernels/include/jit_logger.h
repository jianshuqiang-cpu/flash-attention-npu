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

#ifndef OPTEST_JIT_LOGGER_H
#define OPTEST_JIT_LOGGER_H

#include <cstdio>

enum class JitLogLevel : int
{
    None = 0,
    Info = 1,
    Debug = 2,
};

/**
 * @brief Return the cached runtime JIT log level from CATLASS_JIT_LOG_LEVEL.
 */
extern "C" int GetJitLogLevel();

/**
 * @brief Convert a JIT log level enum to a stable printable label.
 */
inline const char* JitLevelStr(JitLogLevel level)
{
    switch (level) {
        case JitLogLevel::Info:
            return "INFO";
        case JitLogLevel::Debug:
            return "DEBUG";
        default:
            return "NONE";
    }
}

#define JIT_LOGE(fmt, ...) JIT_LOG(JitLogLevel::Info, fmt, ##__VA_ARGS__)

#define JIT_LOG(level_, fmt, ...)                                                         \
    do {                                                                                  \
        if (GetJitLogLevel() >= static_cast<int>(level_)) {                               \
            fprintf(stderr, "[JIT][%s] %s:%d] " fmt "\n", JitLevelStr(level_), __FILE__,  \
                __LINE__, ##__VA_ARGS__);                                                 \
        }                                                                                 \
    } while (0)

#endif // OPTEST_JIT_LOGGER_H
