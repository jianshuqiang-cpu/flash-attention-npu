add_kernel(NAME gemm
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/gemm_impl.cpp)
