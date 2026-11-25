add_kernel(NAME ascend950_matmul_full_dequant
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_full_dequant.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_full_dequant_impl.cpp)
