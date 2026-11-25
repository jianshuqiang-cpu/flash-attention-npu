add_kernel(NAME grouped_matmul_slice_k_per_token_dequant
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_k_per_token_dequant.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_slice_k_per_token_dequant_impl.cpp)
