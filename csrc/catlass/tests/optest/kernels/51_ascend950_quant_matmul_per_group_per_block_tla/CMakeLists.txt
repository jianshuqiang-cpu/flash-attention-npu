add_kernel(NAME ascend950_quant_matmul_per_group_per_block_tla
    NPU_ARCH_LIST 3510
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_per_group_per_block.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_per_group_per_block_impl.cpp)
