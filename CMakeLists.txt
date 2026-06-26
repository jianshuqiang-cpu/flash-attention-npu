# Main CMakeLists.txt for Flash Attention NPU (Ascend 2201 only)
cmake_minimum_required(VERSION 3.18)
project(flash_attn_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_FLASH_ATTN_V3 "Build Flash Attention v3" ON)

if(BUILD_FLASH_ATTN_V3)
    if(EXISTS ${PROJECT_SOURCE_DIR}/csrc/flash_attn_npu_v3)
        add_subdirectory(csrc/flash_attn_npu_v3)
    endif()
endif()
