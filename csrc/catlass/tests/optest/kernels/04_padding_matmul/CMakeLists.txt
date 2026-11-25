add_kernel(NAME padding_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/padding_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/padding_matmul_impl.cpp)
