add_kernel(NAME grouped_matmul_slice_m_gelu
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_gelu.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_gelu_impl.cpp)
