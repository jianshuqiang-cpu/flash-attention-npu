add_kernel(NAME ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_per_tensor_per_channel_dequant.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_per_tensor_per_channel_dequant_impl.cpp)
