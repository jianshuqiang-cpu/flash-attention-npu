# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_source_files_properties(dual_level_quant_mx_batch_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(63_ascend950_dual_level_quant_mx_batch_matmul cube dual_level_quant_mx_batch_matmul.cpp)
target_compile_definitions(63_ascend950_dual_level_quant_mx_batch_matmul PRIVATE L2_CACHE_HINT)
