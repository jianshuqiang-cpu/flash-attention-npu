add_kernel(NAME small_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/small_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/small_matmul_impl.cpp)
