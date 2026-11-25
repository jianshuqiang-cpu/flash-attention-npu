add_kernel(NAME matmul_bias
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_bias.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_bias_impl.cpp)
