add_kernel(NAME matmul_gelu
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gelu.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gelu_impl.cpp)
