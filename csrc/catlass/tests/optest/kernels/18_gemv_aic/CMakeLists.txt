add_kernel(NAME gemv_aic
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/gemv_aic.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/gemv_aic_impl.cpp)
