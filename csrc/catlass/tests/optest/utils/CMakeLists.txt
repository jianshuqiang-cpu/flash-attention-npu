# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(TEST_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================
# libcatlass_kernel_utils.a — Pure ACL type utilities (no torch)
# ============================================================
add_library(catlass_kernel_utils STATIC
    kernel_utils.cpp
)

target_include_directories(catlass_kernel_utils PUBLIC
    ${TEST_UTILS_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
)

target_link_directories(catlass_kernel_utils PUBLIC
    ${ASCEND_HOME_PATH}/lib64
)

install(TARGETS catlass_kernel_utils DESTINATION lib)

# ============================================================
# libcatlass_torch_utils.a — Torch-dependent utilities
# ============================================================
add_library(catlass_torch_utils STATIC
    torch_utils.cpp
)

target_include_directories(catlass_torch_utils PUBLIC
    ${TEST_UTILS_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${CATLASS_DIR}/include
)

target_link_directories(catlass_torch_utils PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${TORCH_NPU_DIR}/lib
)

target_link_libraries(catlass_torch_utils PUBLIC
    catlass_kernel_utils
    ${TORCH_LIBRARIES}
    torch_npu
)

install(TARGETS catlass_torch_utils DESTINATION lib)
