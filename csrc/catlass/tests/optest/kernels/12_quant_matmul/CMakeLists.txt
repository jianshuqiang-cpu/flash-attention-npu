add_kernel(NAME quant_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_impl.cpp)
