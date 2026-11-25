add_kernel(NAME matmul_add
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_add.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_add_impl.cpp)
