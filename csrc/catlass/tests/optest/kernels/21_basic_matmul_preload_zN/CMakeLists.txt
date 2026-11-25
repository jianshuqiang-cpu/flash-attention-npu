add_kernel(NAME basic_matmul_preload_zN
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_preload_zN.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_preload_zN_impl.cpp)
