# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if ("${CATLASS_LIBRARY_KERNELS}" STREQUAL "")
    message(FATAL_ERROR "CATLASS_LIBRARY_KERNELS is not allowed to be empty")
else()
    message(STATUS "CATLASS_LIBRARY_KERNELS=${CATLASS_LIBRARY_KERNELS}")
endif()

set(A2_SUPPORTED_KERNELS
    "00_basic_matmul"
    "08_grouped_matmul"
    "02_grouped_matmul_slice_m"
    "06_optimized_matmul_without_padding"
    "06_optimized_matmul_padding_ab"
    "06_optimized_matmul_padding_a_only"
    "06_optimized_matmul_padding_b_only"
    "12_quant_matmul"
)

set(ASCEND950_SUPPORTED_KERNELS
    "43_ascend950_basic_matmul"
)

set(CATLASS_KERNELS_ARCH "")

if ("${CATLASS_ARCH}" STREQUAL "2201")
    set(CATLASS_KERNELS_ARCH "a2")
elseif ("${CATLASS_ARCH}" STREQUAL "3510")
    set(CATLASS_KERNELS_ARCH "Ascend950")
else ()
    message(FATAL_ERROR "unsupported CATLASS_ARCH=${CATLASS_ARCH}")
endif()

foreach(item IN LISTS ASCEND950_SUPPORTED_KERNELS)
    string(FIND ${CATLASS_LIBRARY_KERNELS} ${item} index)
    string(FIND ${item} ${CATLASS_LIBRARY_KERNELS} index2)
    if(NOT index EQUAL -1 OR NOT index2 EQUAL -1)
        if (NOT "${CATLASS_ARCH}" STREQUAL "3510")
            message(FATAL_ERROR "unsupported CATLASS_ARCH '${CATLASS_ARCH}' for "
                                 "kernel '${item}' which is matched by "
                                 "CATLASS_LIBRARY_KERNELS '${CATLASS_LIBRARY_KERNELS}'")
        endif()
    endif()
endforeach()

foreach(item IN LISTS A2_SUPPORTED_KERNELS)
    string(FIND ${CATLASS_LIBRARY_KERNELS} ${item} index)
    string(FIND ${item} ${CATLASS_LIBRARY_KERNELS} index2)
    if(NOT index EQUAL -1 OR NOT index2 EQUAL -1)
        if (NOT "${CATLASS_ARCH}" STREQUAL "2201")
            message(FATAL_ERROR "unsupported CATLASS_ARCH '${CATLASS_ARCH}' for "
                                 "kernel '${item}' which is matched by "
                                 "CATLASS_LIBRARY_KERNELS '${CATLASS_LIBRARY_KERNELS}'")
        endif()
    endif()
endforeach()

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/code_generator.py
        --kernels ${CATLASS_LIBRARY_KERNELS}
        --workspace-dir ${CMAKE_CURRENT_BINARY_DIR}
        --arch ${CATLASS_KERNELS_ARCH}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE CATLASS_LIBRARY_CODE_GENERATION_RESULT
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log
)

if(NOT CATLASS_LIBRARY_CODE_GENERATION_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate catlass library operator code. "
                        "See details in ${CMAKE_CURRENT_BINARY_DIR}/catlass_library_code_generation.log")
endif()

file(GLOB_RECURSE CATLASS_LIBRARY_GENERATED_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated/*.cpp)

message(STATUS "Catlass library operator code was generated successfully.")

list(APPEND CATLASS_LIBRARY_GENERATED_SRCS src/manifest.cpp)

set_source_files_properties(${CATLASS_LIBRARY_GENERATED_SRCS} PROPERTIES LANGUAGE ASC)

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${CATLASS_ARCH}>)
add_compile_definitions(CATLASS_ARCH=${CATLASS_ARCH})

# common compile options
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
    # these options may be removed after CANN 8.5.0
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
)

# these options may be removed after CANN 8.5.0
add_link_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>)

add_library(catlass_kernels SHARED
    ${CATLASS_LIBRARY_GENERATED_SRCS}
    src/manifest.cpp
)

set_target_properties(catlass_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(catlass_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/asc/include
)

target_compile_definitions(catlass_kernels PRIVATE L2_CACHE_HINT)
target_link_directories(catlass_kernels PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(catlass_kernels PRIVATE
    # AscendC necessary libraries
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl m
    # other necessary libraries
    tiling_api nnopbase platform
    # system libraries
    dl
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../output CACHE PATH "Installation prefix for CATLASS" FORCE)

install(TARGETS catlass_kernels
        DESTINATION lib64
        COMPONENT catlass_kernels)
