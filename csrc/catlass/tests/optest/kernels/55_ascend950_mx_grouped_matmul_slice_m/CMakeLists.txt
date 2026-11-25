add_kernel(NAME ascend950_mx_grouped_matmul_slice_m
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_grouped_matmul_slice_m.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/mx_grouped_matmul_slice_m_impl.cpp)
