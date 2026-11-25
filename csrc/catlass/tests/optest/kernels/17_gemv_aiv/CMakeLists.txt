add_kernel(NAME gemv_aiv
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/gemv_aiv.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/gemv_aiv_impl.cpp)
