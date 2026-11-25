# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_MXFP8_FAI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend950_fp8_mx_flash_attention_infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tiling/fai_tiling.cpp
)
set_source_files_properties(${_MXFP8_FAI_SOURCES} PROPERTIES LANGUAGE ASC)

# dav-3510 msSanitizer is incompatible with the mixed host/device translation unit in this kernel.
add_kernel(
    NAME ascend950_fp8_mx_flash_attention_infer
    KERNEL_TYPE prebuilt
    NPU_ARCH_LIST 3510
    SKIP_MS_BUILD
    ${_MXFP8_FAI_SOURCES}
)

foreach(_SRC ${_MXFP8_FAI_SOURCES})
    foreach(_ARCH 3510)
        set(_LIB_NAME "catlass_kernel_${_ARCH}_ascend950_fp8_mx_flash_attention_infer")
        if(TARGET ${_LIB_NAME})
            target_compile_options(${_LIB_NAME} PRIVATE
                "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-record-overflow=true>"
                "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
            )
            target_include_directories(${_LIB_NAME} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/tiling
                ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            )
        endif()
    endforeach()
endforeach()
