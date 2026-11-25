# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

import re
import acl
import pytest
import torch_npu

def get_npu_arch():
    """
    Acquire NPU_ARCH tag from current device
    """
    device_name = acl.get_soc_name()

    if re.match(r"Ascend910B.+", device_name, re.I) or re.search(
        r"Ascend910_93", device_name, re.I
    ):
        return 2201
    elif re.search("Ascend950(PR|DT)", device_name, re.I):
        return 3510
    else:
        raise ValueError(f"Unsupported device name: {device_name}")

def only_on_npu_arch(npu_arch: int):
    return pytest.mark.skipif(
        torch_npu.npu.device_count() <= 0 or npu_arch != get_npu_arch(), 
        reason="torch-catlass integration tests require an available Ascend NPU " + \
        f"and this can only runs on {npu_arch}",
    )

only_on_2201 = only_on_npu_arch(2201)
only_on_3510 = only_on_npu_arch(3510)