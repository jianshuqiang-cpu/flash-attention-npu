add_kernel(NAME streamk_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/streamk_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/streamk_matmul_impl.cpp)
