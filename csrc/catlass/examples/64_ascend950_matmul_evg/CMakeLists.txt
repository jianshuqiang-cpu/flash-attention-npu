# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(matmul_evg_add.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_add mix matmul_evg_add.cpp)

set_source_files_properties(matmul_evg_leaky_relu.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_leaky_relu mix matmul_evg_leaky_relu.cpp)

set_source_files_properties(matmul_evg_sigmoid.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_sigmoid mix matmul_evg_sigmoid.cpp)
target_link_libraries(64_ascend950_matmul_evg_sigmoid PRIVATE m)

set_source_files_properties(matmul_evg_silu.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_silu mix matmul_evg_silu.cpp)
target_link_libraries(64_ascend950_matmul_evg_silu PRIVATE m)

set_source_files_properties(matmul_evg_tanh.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_tanh mix matmul_evg_tanh.cpp)
target_link_libraries(64_ascend950_matmul_evg_tanh PRIVATE m)

set_source_files_properties(matmul_evg_bias.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_bias mix matmul_evg_bias.cpp)

set_source_files_properties(matmul_evg_add_ub.cpp PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(64_ascend950_matmul_evg_add_ub mix matmul_evg_add_ub.cpp)

# 一次编译本目录全部 7 个可执行文件（install 仍需对各 target 分别执行 build.sh）
add_custom_target(64_ascend950_matmul_evg_all)
add_dependencies(64_ascend950_matmul_evg_all
    64_ascend950_matmul_evg_add
    64_ascend950_matmul_evg_leaky_relu
    64_ascend950_matmul_evg_sigmoid
    64_ascend950_matmul_evg_silu
    64_ascend950_matmul_evg_tanh
    64_ascend950_matmul_evg_bias
    64_ascend950_matmul_evg_add_ub
)
