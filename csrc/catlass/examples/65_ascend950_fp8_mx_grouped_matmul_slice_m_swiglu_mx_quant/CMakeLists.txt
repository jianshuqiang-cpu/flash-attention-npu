# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_source_files_properties(fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant cube fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant.cpp)
target_compile_definitions(65_ascend950_fp8_mx_grouped_matmul_slice_m_swiglu_mx_quant PRIVATE L2_CACHE_HINT)
