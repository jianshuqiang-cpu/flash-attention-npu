# CMakeLists.txt for flash_attn_npu_v3 (Ascend 2201)
cmake_minimum_required(VERSION 3.18)
project(flash_attn_npu_v3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME $ENV{ASCEND_HOME})
else()
    set(ASCEND_HOME "/usr/local/Ascend")
endif()

# Detect system architecture (x86_64 or aarch64)
if(NOT DEFINED ASCEND_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(ASCEND_ARCH "aarch64-linux")
    else()
        set(ASCEND_ARCH "x86_64-linux")
    endif()
endif()

message(STATUS "Using Ascend toolkit from: ${ASCEND_HOME}")
message(STATUS "System architecture: ${ASCEND_ARCH}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(PREPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include/interface
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/impl/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${ASCEND_ARCH}/tikcpp/tikcfw
    ${ASCEND_HOME}/${ASCEND_ARCH}/include
    ${ASCEND_HOME}/${ASCEND_ARCH}/include/ascendc/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/ascendc/include/basic_api
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/pkg_inc
    ${ASCEND_HOME}/pkg_inc/profiling
    ${ASCEND_HOME}/runtime/include
    ${ASCEND_HOME}/include/experiment/runtime
    ${ASCEND_HOME}/include/experiment/msprof
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${PROJECT_SOURCE_DIR}/../catlass/include
)

link_directories(
    ${ASCEND_HOME}/compiler/lib64
    ${ASCEND_HOME}/aarch64-linux/lib64
    ${ASCEND_HOME}/lib64
    ${TORCH_NPU_PATH}/lib
)

add_library(pch_target INTERFACE)
target_include_directories(pch_target INTERFACE
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/include/interface
    ${ASCEND_HOME}/${ASCEND_ARCH}/asc/impl/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${ASCEND_ARCH}/tikcpp/tikcfw
    ${ASCEND_HOME}/${ASCEND_ARCH}/include
    ${ASCEND_HOME}/${ASCEND_ARCH}/include/ascendc/basic_api
    ${ASCEND_HOME}/${ASCEND_ARCH}/ascendc/include/basic_api
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/pkg_inc
    ${ASCEND_HOME}/pkg_inc/profiling
    ${ASCEND_HOME}/runtime/include
    ${ASCEND_HOME}/include/experiment/runtime
    ${ASCEND_HOME}/include/experiment/msprof
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${PROJECT_SOURCE_DIR}/../catlass/include
)
target_precompile_headers(pch_target INTERFACE ${PROJECT_SOURCE_DIR}/pch.hpp)

add_library(kernel_inst_64 OBJECT kernel_inst_64.cpp)
target_link_libraries(kernel_inst_64 PRIVATE pch_target)
target_compile_definitions(kernel_inst_64 PRIVATE CATLASS_ARCH=2201)

add_library(kernel_inst_128 OBJECT kernel_inst_128.cpp)
target_link_libraries(kernel_inst_128 PRIVATE pch_target)
target_compile_definitions(kernel_inst_128 PRIVATE CATLASS_ARCH=2201)

add_library(kernel_inst_192 OBJECT kernel_inst_192.cpp)
target_link_libraries(kernel_inst_192 PRIVATE pch_target)
target_compile_definitions(kernel_inst_192 PRIVATE CATLASS_ARCH=2201)

add_library(kernel_inst_256 OBJECT kernel_inst_256.cpp)
target_link_libraries(kernel_inst_256 PRIVATE pch_target)
target_compile_definitions(kernel_inst_256 PRIVATE CATLASS_ARCH=2201)

add_library(flash_attn_npu_3 SHARED
    flash_api.cpp
    $<TARGET_OBJECTS:kernel_inst_64>
    $<TARGET_OBJECTS:kernel_inst_128>
    $<TARGET_OBJECTS:kernel_inst_192>
    $<TARGET_OBJECTS:kernel_inst_256>
)

target_link_libraries(flash_attn_npu_3
    PRIVATE
    pch_target
    ascendcl
    torch_npu
    tiling_api
    platform
    ${TORCH_LIBRARIES}
)

target_compile_definitions(flash_attn_npu_3
    PRIVATE
    CATLASS_ARCH=2201
)

set_target_properties(flash_attn_npu_3 PROPERTIES
    PREFIX ""
    OUTPUT_NAME "flash_attn_npu_3"
)

install(TARGETS flash_attn_npu_3
    LIBRARY DESTINATION .
)
