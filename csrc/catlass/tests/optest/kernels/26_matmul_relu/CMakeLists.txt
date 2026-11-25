add_kernel(NAME matmul_relu
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_relu.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_relu_impl.cpp)
