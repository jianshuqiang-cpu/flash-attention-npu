# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

"""版本号管理模块，自动生成带时间戳的版本号"""

from datetime import datetime

# 基础版本号
BASE_VERSION = "0.1.0"


def get_version():
    """
    生成带时间戳的版本号

    格式: {BASE_VERSION}+{YYYYMMDDHHMMSS}
    例如: 0.1.0+20240101120000
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{BASE_VERSION}+{timestamp}"


# 在构建时生成版本号
__version__ = get_version()
