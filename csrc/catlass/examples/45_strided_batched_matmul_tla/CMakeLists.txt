# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(strided_batched_matmul_tla.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(45_strided_batched_matmul_tla cube strided_batched_matmul_tla.cpp)

function(add_variant target_name)
    catlass_example_add_executable(${target_name} cube strided_batched_matmul_tla.cpp)
    target_compile_definitions(${target_name} PRIVATE L2_CACHE_HINT ${ARGN})
endfunction()


# The batched matrix A is transposed as the logical shape `(M, B, K)`
add_variant(45_strided_transposed_batched_matmul_tla ENABLE_TRANS_A)
