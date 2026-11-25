add_kernel(NAME splitk_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/splitk_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/splitk_matmul_impl.cpp)
