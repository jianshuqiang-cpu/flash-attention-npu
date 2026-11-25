# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(get_cann_version_and_soc_name)
    set(GET_CANN_VERSION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/get_cann_version_and_soc_name.cpp)
    set(GET_CANN_VERSION_BIN ${CMAKE_BINARY_DIR}/get_cann_version_and_soc_name)

    execute_process(
        COMMAND g++ ${GET_CANN_VERSION_SRC} -o ${GET_CANN_VERSION_BIN} -std=c++17
            -I${ASCEND_HOME_PATH}/include -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lc_sec
        RESULT_VARIABLE COMPILE_RESULT
        OUTPUT_VARIABLE COMPILE_OUTPUT
        ERROR_VARIABLE COMPILE_ERROR
    )

    if(NOT COMPILE_RESULT EQUAL 0)
        message(WARNING "Failed to compile get_cann_version_and_soc_name: ${COMPILE_ERROR}")
        return()
    endif()

    execute_process(
        COMMAND ${GET_CANN_VERSION_BIN}
        RESULT_VARIABLE RUN_RESULT
        OUTPUT_VARIABLE RUN_OUTPUT
        ERROR_VARIABLE RUN_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT RUN_RESULT EQUAL 0)
        message(WARNING "Failed to run get_cann_version_and_soc_name: ${RUN_ERROR}")
        return()
    endif()

    string(REGEX MATCH "^[^ ]+" CANN_VERSION_TMP "${RUN_OUTPUT}")

    # 只在调用方未预先设定 CANN_VERSION 时才覆盖
    if(NOT DEFINED CANN_VERSION OR CANN_VERSION STREQUAL "")
        set(CANN_VERSION ${CANN_VERSION_TMP} PARENT_SCOPE)
        message("Detected CANN_VERSION: ${CANN_VERSION_TMP}")
    else()
        message("Using preset CANN_VERSION: ${CANN_VERSION}")
    endif()

    string(LENGTH "${RUN_OUTPUT}" OUTPUT_LEN)
    string(FIND "${RUN_OUTPUT}" " " SPACE_POS)
    if(SPACE_POS GREATER_EQUAL 0)
        math(EXPR SOC_NAME_START "${SPACE_POS} + 1")
        string(SUBSTRING "${RUN_OUTPUT}" ${SOC_NAME_START} -1 NPU_MODEL_TMP)

        # 只在调用方未预先设定 NPU_MODEL 时才覆盖
        if(NOT DEFINED NPU_MODEL OR NPU_MODEL STREQUAL "")
            set(NPU_MODEL ${NPU_MODEL_TMP} PARENT_SCOPE)
            message("Detected NPU_MODEL: ${NPU_MODEL_TMP}")
        else()
            message("Using preset NPU_MODEL: ${NPU_MODEL}")
        endif()
    endif()
endfunction()
