add_kernel(NAME w4a4_matmul_per_token_per_channel_dequant
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/w4a4_matmul_per_token_per_channel_dequant.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/w4a4_matmul_per_token_per_channel_dequant_impl.cpp)
