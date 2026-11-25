add_kernel(NAME single_core_splitk_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/single_core_splitk_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/single_core_splitk_matmul_impl.cpp)
