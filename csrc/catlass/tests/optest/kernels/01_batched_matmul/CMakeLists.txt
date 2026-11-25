add_kernel(NAME batched_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/batched_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/batched_matmul_impl.cpp)
