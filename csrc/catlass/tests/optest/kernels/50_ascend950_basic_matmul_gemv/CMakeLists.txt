add_kernel(NAME ascend950_basic_matmul_gemv
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_gemv.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/basic_matmul_gemv_impl.cpp)
