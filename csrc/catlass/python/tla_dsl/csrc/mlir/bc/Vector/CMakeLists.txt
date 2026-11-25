if (BISHENGIR_BUILD_TEMPLATE STREQUAL "ON")
  compile_cpp_to_bc("aiv" "c310")
  compile_cpp_to_bc("mix_aiv" "c310")
endif()
