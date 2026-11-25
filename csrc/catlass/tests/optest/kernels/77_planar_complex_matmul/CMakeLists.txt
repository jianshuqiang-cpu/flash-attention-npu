add_kernel(NAME planar_complex_matmul
    NPU_ARCH_LIST 2201
    KERNEL_TYPE jit
    ${CMAKE_CURRENT_SOURCE_DIR}/planar_complex_matmul.cpp
    TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/planar_complex_matmul_impl.cpp)
