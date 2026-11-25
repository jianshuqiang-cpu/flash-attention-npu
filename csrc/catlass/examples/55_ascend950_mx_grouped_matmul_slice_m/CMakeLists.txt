# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_source_files_properties(mx_grouped_matmul_slice_m.cpp PROPERTIES LANGUAGE ASC)

function(add_mx_grouped_matmul_variant target_name)
    catlass_example_add_executable(${target_name} cube mx_grouped_matmul_slice_m.cpp)
    target_compile_definitions(${target_name} PRIVATE L2_CACHE_HINT ${ARGN})
endfunction()

add_mx_grouped_matmul_variant(55_ascend950_mx_grouped_matmul_slice_m)
add_mx_grouped_matmul_variant(55_ascend950_mx_grouped_matmul_slice_m_aswt
    MX_GMM_ENABLE_ASWT=1)
add_mx_grouped_matmul_variant(55_ascend950_mx_grouped_matmul_slice_m_basem
    MX_GMM_ENABLE_ASWT=1
    MX_GMM_ENABLE_BASE_M=1)
# Keep MX_GMM_PRELOAD_STAGES at its default 1 unless the preload BlockMmad
# L1 stage/event budget is changed together. See the macro comment in
# mx_grouped_matmul_slice_m.cpp before tuning it.
add_mx_grouped_matmul_variant(55_ascend950_mx_grouped_matmul_slice_m_preload
    MX_GMM_ENABLE_PRELOAD=1)
add_mx_grouped_matmul_variant(55_ascend950_mx_grouped_matmul_slice_m_all
    MX_GMM_ENABLE_ASWT=1
    MX_GMM_ENABLE_BASE_M=1
    MX_GMM_ENABLE_PRELOAD=1)
