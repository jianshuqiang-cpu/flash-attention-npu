add_kernel(NAME ascend950_matmul_fixpipe_opti
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fixpipe_opti.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fixpipe_opti_impl.cpp)
