add_kernel(NAME matmul_silu
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_silu.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_silu_impl.cpp)
