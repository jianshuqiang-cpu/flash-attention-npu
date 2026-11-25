add_kernel(NAME grouped_matmul_slice_m_per_token_dequant_moe
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_per_token_dequant_moe.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_m_per_token_dequant_moe_impl.cpp)
