add_kernel(NAME basic_matmul_tla
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_tla.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_tla_impl.cpp)
