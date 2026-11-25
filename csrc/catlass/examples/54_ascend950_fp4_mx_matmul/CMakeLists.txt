# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_source_files_properties(fp4_mx_matmul.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(54_ascend950_fp4_mx_matmul cube fp4_mx_matmul.cpp)
target_compile_definitions(54_ascend950_fp4_mx_matmul PRIVATE L2_CACHE_HINT)

set_source_files_properties(fp4_mx_matmul_aswt.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(54_ascend950_fp4_mx_matmul_aswt cube fp4_mx_matmul_aswt.cpp)
target_compile_definitions(54_ascend950_fp4_mx_matmul_aswt PRIVATE L2_CACHE_HINT)
