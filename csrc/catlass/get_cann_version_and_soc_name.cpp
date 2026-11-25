/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>

#include <acl/acl.h>
#include <securec.h>

#define CANN_VERSION_BUFFER_LEN 100
#define CANN_VERSION_BUFFER_FMT "%99[^/]"

int main()
{
    const char* ascend_home_path = std::getenv("ASCEND_HOME_PATH");
    if (!ascend_home_path) {
        fprintf(stderr, "[ERROR] ASCEND_HOME_PATH is not set\n");
        return -1;
    }

    const char* start = strstr(ascend_home_path, "/cann-");
    if (!start) {
        fprintf(stderr, "[ERROR] ASCEND_HOME_PATH does not contain CANN version\n");
        return -1;
    }

    start += 6; // skip "/cann-"
    char version[CANN_VERSION_BUFFER_LEN] = {0};
    int ret = sscanf_s(start, CANN_VERSION_BUFFER_FMT, version, CANN_VERSION_BUFFER_LEN);
    if (ret == -1) {
        fprintf(stderr, "[ERROR] Parsing CANN version failed\n");
        return -1;
    }

    const char* socName = aclrtGetSocName();
    if (socName) {
        fprintf(stdout, "%s %s\n", version, socName);
    } else {
        fprintf(stdout, "%s\n", version);
        fprintf(stderr, "[ERROR] aclrtGetSocName failed\n"); // cmake only gets stdout
    }
    return 0;
}
