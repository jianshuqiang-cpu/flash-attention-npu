# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(GOOGLETEST_INCLUDE_DIR ${CATLASS_3RDPARTY_DIR}/googletest/googletest/include)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${CATLASS_INCLUDE_DIR}
)

link_directories(${ASCEND_HOME_PATH}/lib64)

# Two separate binaries, one per architecture, to avoid ODR violation on
# arch-dependent structs (e.g. Nd2NzParams differs between 2201 / 3510).
add_executable(catlass_unittest_2201 catlass_unittest.cpp)
add_executable(catlass_unittest_3510 catlass_unittest.cpp)

set_target_properties(catlass_unittest_2201 catlass_unittest_3510 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

function(testcase MODE ARCH)
    set(SOURCES ${ARGN})

    foreach(SRC ${SOURCES})
        get_filename_component(FILE_NAME ${SRC} NAME_WE)

        if(MODE STREQUAL NPU)
            add_library(${FILE_NAME}_npu OBJECT ${SRC})
            set_source_files_properties(${SRC} PROPERTIES LANGUAGE ASC)
            target_compile_options(${FILE_NAME}_npu PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${ARCH}>)
            target_sources(catlass_unittest_${ARCH} PRIVATE $<TARGET_OBJECTS:${FILE_NAME}_npu>)
        endif()

        if(MODE STREQUAL STUB)
            add_library(${FILE_NAME}_stub_${ARCH} OBJECT ${SRC})
            set_target_properties(${FILE_NAME}_stub_${ARCH} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
            target_include_directories(${FILE_NAME}_stub_${ARCH} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stub)
            target_compile_definitions(${FILE_NAME}_stub_${ARCH} PRIVATE CATLASS_ARCH=${ARCH} __NPU_ARCH__=${ARCH})
            target_sources(catlass_unittest_${ARCH} PRIVATE $<TARGET_OBJECTS:${FILE_NAME}_stub_${ARCH}>)
        endif()
    endforeach()
endfunction()

################ COVERAGE ################
if(DEFINED ENABLE_COVERAGE AND ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-inline -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
    message(STATUS "Coverage reporting enabled")

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(LCOV lcov REQUIRED)
    find_program(GCOV gcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)

    set(COVERAGE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/coverage)
    set(COVERAGE_INFO_FILE ${COVERAGE_OUTPUT_DIR}/coverage.info)
    set(COVERAGE_HTML_DIR ${COVERAGE_OUTPUT_DIR}/html)

    add_custom_target(coverage_collect
        COMMAND ${CMAKE_COMMAND} -E make_directory ${COVERAGE_OUTPUT_DIR}
        COMMAND ${LCOV} --capture --directory ${CMAKE_BINARY_DIR} --output-file ${COVERAGE_INFO_FILE}.raw --gcov-tool ${GCOV}
        COMMAND ${LCOV} --remove ${COVERAGE_INFO_FILE}.raw
            '*/tests/*'
            '*/3rdparty/*'
            '/usr/*'
            --output-file ${COVERAGE_INFO_FILE} --ignore-errors unused
        COMMAND ${GENHTML} ${COVERAGE_INFO_FILE} --output-directory ${COVERAGE_HTML_DIR}
        COMMAND ${LCOV} --list ${COVERAGE_INFO_FILE}
        COMMENT "Generating coverage report"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
################ COVERAGE ################

file(GLOB CATLASS_UNITTEST_SRCS_TILE_STUB ${CMAKE_CURRENT_SOURCE_DIR}/catlass/gemm/tile/*.cpp)
testcase(STUB 2201 ${CATLASS_UNITTEST_SRCS_TILE_STUB})
testcase(STUB 3510 ${CATLASS_UNITTEST_SRCS_TILE_STUB})

# Aggregative target for `--tests catlass_unittest`
add_custom_target(catlass_unittest DEPENDS catlass_unittest_2201 catlass_unittest_3510)

foreach(exe catlass_unittest_2201 catlass_unittest_3510)
    target_link_libraries(${exe} PRIVATE
        gtest gtest_main m pthread
        ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl
        tiling_api nnopbase platform dl
    )
    if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
        target_link_libraries(${exe} PRIVATE ascendalog)
    endif()
    if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
        target_link_libraries(${exe} PRIVATE unified_dlog)
    endif()
endforeach()
