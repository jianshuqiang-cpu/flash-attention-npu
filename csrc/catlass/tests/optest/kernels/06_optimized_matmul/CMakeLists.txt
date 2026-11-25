add_kernel(NAME optimized_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/optimized_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/optimized_matmul_impl.cpp)
