add_kernel(NAME matmul_full_loadA
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_full_loadA.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/matmul_full_loadA_impl.cpp)
