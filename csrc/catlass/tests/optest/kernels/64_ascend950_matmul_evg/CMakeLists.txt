# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_EVG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(_EVG_TEMPLATES
    ${_EVG_DIR}/matmul_evg_add_impl.cpp
    ${_EVG_DIR}/matmul_evg_add_ub_impl.cpp
    ${_EVG_DIR}/matmul_evg_bias_impl.cpp
    ${_EVG_DIR}/matmul_evg_leaky_relu_impl.cpp
    ${_EVG_DIR}/matmul_evg_sigmoid_impl.cpp
    ${_EVG_DIR}/matmul_evg_silu_impl.cpp
    ${_EVG_DIR}/matmul_evg_tanh_impl.cpp
)

add_kernel(NAME matmul_evg
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${_EVG_DIR}/matmul_evg.cpp
    TEMPLATE ${_EVG_TEMPLATES})
