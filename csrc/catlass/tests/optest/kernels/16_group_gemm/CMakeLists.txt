add_kernel(NAME group_gemm
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/group_gemm.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/group_gemm_impl.cpp)
