# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(_PFA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_flash_attention_chunk_prefill.cpp
)
set_source_files_properties(${_PFA_SOURCES} PROPERTIES LANGUAGE ASC)

add_kernel(NAME ascend950_flash_attention_chunk_prefill
    NPU_ARCH_LIST 3510
    KERNEL_TYPE prebuilt
    ${_PFA_SOURCES}
)

foreach(_SRC ${_PFA_SOURCES})
    get_filename_component(_SRC_NAME ${_SRC} NAME)
    foreach(_ARCH 3510)
        foreach(_MS_SUFFIX "" "_ms")
            set(_LIB_NAME "catlass_kernel_${_ARCH}_ascend950_flash_attention_chunk_prefill${_MS_SUFFIX}")
            if(TARGET ${_LIB_NAME})
                target_compile_definitions(${_LIB_NAME} PRIVATE TILING_KEY_VAR)
                target_include_directories(${_LIB_NAME} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                )
            endif()
        endforeach()
    endforeach()
endforeach()
