cmake_minimum_required(VERSION 3.20)
# 默认用宿主 sysroot(conda 自带 glibc 2.28 缺 LLVM 所需符号);显式传入时保留
if(NOT DEFINED CMAKE_SYSROOT OR "${CMAKE_SYSROOT}" STREQUAL "")
  set(CMAKE_SYSROOT "/")
endif()
project(tla LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")
option(BISHENGIR_BUILD_TEMPLATE "Build C310 HIVM template bitcode." ON)

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
  message(FATAL_ERROR "ASCEND_HOME_PATH is not defined.")
endif()
message("-- ASCEND_HOME_PATH Value: ${ASCEND_HOME_PATH}")

# Catlass public headers for bisheng template bitcode (see mlir/bc).
if(DEFINED CATLASS_INCLUDE_DIR AND NOT "${CATLASS_INCLUDE_DIR}" STREQUAL "")
  set(CATLASS_DSL_CATLASS_INCLUDE_DIR "${CATLASS_INCLUDE_DIR}")
else()
  get_filename_component(CATLASS_DSL_CATLASS_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include" ABSOLUTE)
endif()
message("-- CATLASS_DSL_CATLASS_INCLUDE_DIR Value: ${CATLASS_DSL_CATLASS_INCLUDE_DIR}")

# AscendNPU-IR lives under python/tla_dsl/3rdparty (mlir is under
# python/tla_dsl/csrc/mlir). The TLA compiler intentionally uses the Dev
# BiShengIR tree for all bishengir headers/libs so pass declarations and linked
# implementations stay from the same checkout.
set(CATLASS_DSL_ASCENDNPU_IR_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/AscendNPU-IR)
if(DEFINED ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR}
   AND NOT "$ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR}" STREQUAL "")
  set(CATLASS_DSL_ASCENDNPU_IR_ROOT "$ENV{CATLASS_DSL_PREBUILT_ASCENDNPU_IR}")
endif()
message("-- CATLASS_DSL_ASCENDNPU_IR_ROOT Value: ${CATLASS_DSL_ASCENDNPU_IR_ROOT}")

set(CATLASS_DSL_HIVM_SOURCE_INCLUDE_DIR
    ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/bishengir/include)
set(CATLASS_DSL_HIVM_GENERATED_INCLUDE_DIR
    ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/tools/bishengir/include)
set(CATLASS_DSL_HIVM_LIB_DIR
    ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/lib)
set(CATLASS_DSL_HIVM_HEADER
    ${CATLASS_DSL_HIVM_SOURCE_INCLUDE_DIR}/bishengir/Dialect/HIVM/IR/HIVM.h)
set(CATLASS_DSL_HIVM_ENUMS_INC
    ${CATLASS_DSL_HIVM_GENERATED_INCLUDE_DIR}/bishengir/Interfaces/BiShengIREnums.h.inc)
set(CATLASS_DSL_HIVM_REQUIRED_LIBS
    MLIRHIVMUtils
    MLIRHIVMDialect
    MLIRHIVMTransforms
    MLIRHIVMToStandard
    MLIRAVEDialect
    MLIRHIVMAVETransforms
    MLIRAVEUtils
    MLIRHIVMAVEToAVEIntrin
    MLIRHIVMAVEToStandard
    MLIRHIVMRegbaseIntrinsDialect
    MLIRHIVMRegbaseIntrinsUtils
    MLIRHFusionToHIVM
    MLIRTensorUtils
    MLIRAnnotationDialect
    BiShengIRDialectUtils
    BiShengIRTensorDialect
    MLIRHFusionDialect
    MLIRHFusionUtils
    MLIRMathExtDialect
    BiShengIRSymbolDialect
    BiShengIRMemRefDialect
    MLIRMemRefExtDialect
    MLIRHACCUtils
    MLIRHACCTransforms
    MLIRHACCDialect
    MLIRHMAPDialect
    BiShengIRScopeDialect)
if(NOT EXISTS ${CATLASS_DSL_HIVM_HEADER} OR NOT EXISTS ${CATLASS_DSL_HIVM_ENUMS_INC})
  message(FATAL_ERROR
          "AscendNPU-IR generated headers are required. Build "
          "python/tla_dsl/3rdparty/AscendNPU-IR first or set CATLASS_DSL_PREBUILT_ASCENDNPU_IR.")
endif()
foreach(tla_dsl_hivm_lib IN LISTS CATLASS_DSL_HIVM_REQUIRED_LIBS)
  file(GLOB tla_dsl_hivm_matches
       "${CATLASS_DSL_HIVM_LIB_DIR}/lib${tla_dsl_hivm_lib}.so"
       "${CATLASS_DSL_HIVM_LIB_DIR}/lib${tla_dsl_hivm_lib}.a"
       "${CATLASS_DSL_HIVM_LIB_DIR}/lib${tla_dsl_hivm_lib}.dylib")
  if("${tla_dsl_hivm_matches}" STREQUAL "")
    message(FATAL_ERROR
            "AscendNPU-IR library ${tla_dsl_hivm_lib} is required. Build "
            "python/tla_dsl/3rdparty/AscendNPU-IR first or set CATLASS_DSL_PREBUILT_ASCENDNPU_IR.")
  endif()
  list(GET tla_dsl_hivm_matches 0 tla_dsl_hivm_match)
  list(APPEND CATLASS_DSL_HIVM_REQUIRED_LIB_FILES "${tla_dsl_hivm_match}")
endforeach()

set(MLIR_DIR ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/install/lib/cmake/mlir)
set(LLVM_DIR ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/install/lib/cmake/llvm)
set(MLIR_TBLGEN_INCLUDE_DIR ${CATLASS_DSL_ASCENDNPU_IR_ROOT}/build/install/include)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bc/CMakeLists.txt")
  add_subdirectory(bc)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)
# python/tla_dsl: sibling of catlass/ (Python pkg) and csrc/ (contains tests/, pyproject, .artifacts, tools).
set(TLA_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${TLA_REPO_ROOT}/.artifacts/pydeps:$ENV{PYTHONPATH}"
            ${Python3_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE CATLASS_DSL_PYBIND11_INCLUDES
    ERROR_VARIABLE CATLASS_DSL_PYBIND11_ERROR
    RESULT_VARIABLE CATLASS_DSL_PYBIND11_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT CATLASS_DSL_PYBIND11_RESULT EQUAL 0 OR CATLASS_DSL_PYBIND11_INCLUDES STREQUAL "")
    message(FATAL_ERROR
            "pybind11 is required to build the Python MLIR bridge. "
            "CMake could not find pybind11::headers and "
            "'${Python3_EXECUTABLE} -m pybind11 --includes' failed. "
            "Install pybind11 in the Python environment used for this build. "
            "pybind11 error: ${CATLASS_DSL_PYBIND11_ERROR}")
  endif()
  separate_arguments(CATLASS_DSL_PYBIND11_INCLUDE_FLAGS UNIX_COMMAND
                     "${CATLASS_DSL_PYBIND11_INCLUDES}")
  set(CATLASS_DSL_PYBIND11_INCLUDE_DIRS)
  foreach(_flag IN LISTS CATLASS_DSL_PYBIND11_INCLUDE_FLAGS)
    if(_flag MATCHES "^-I(.+)$")
      list(APPEND CATLASS_DSL_PYBIND11_INCLUDE_DIRS "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(CATLASS_DSL_PYBIND11_INCLUDE_DIRS STREQUAL "")
    message(FATAL_ERROR
            "pybind11 is required to build the Python MLIR bridge. "
            "'${Python3_EXECUTABLE} -m pybind11 --includes' produced no include directories.")
  endif()
endif()

set(TLA_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/include/Dialect/Tla/IR/Tla.td)
# Dialect TableGen targets (Tla dialect).
find_package(MLIR REQUIRED CONFIG)
# Some MLIR installations do not expose MLIR_TBLGEN_INCLUDE_DIR; mlir-tblgen
# still needs the installed TableGen include tree (fall back to MLIR_INCLUDE_DIRS).
if(NOT DEFINED MLIR_TBLGEN_INCLUDE_DIR OR "${MLIR_TBLGEN_INCLUDE_DIR}" STREQUAL "")
  if(DEFINED MLIR_INCLUDE_DIRS AND NOT "${MLIR_INCLUDE_DIRS}" STREQUAL "")
    set(MLIR_TBLGEN_INCLUDE_DIR "${MLIR_INCLUDE_DIRS}")
  elseif(DEFINED LLVM_INCLUDE_DIRS AND NOT "${LLVM_INCLUDE_DIRS}" STREQUAL "")
    set(MLIR_TBLGEN_INCLUDE_DIR "${LLVM_INCLUDE_DIRS}")
  endif()
endif()
if(NOT DEFINED MLIR_TBLGEN_INCLUDE_DIR OR "${MLIR_TBLGEN_INCLUDE_DIR}" STREQUAL "")
  message(FATAL_ERROR
          "MLIR_TBLGEN_INCLUDE_DIR is not set and could not be inferred. "
          "Pass -DMLIR_TBLGEN_INCLUDE_DIR=<dir containing mlir/IR/OpBase.td>.")
endif()
message(STATUS "MLIR_TBLGEN_INCLUDE_DIR=${MLIR_TBLGEN_INCLUDE_DIR}")
find_file(CATLASS_DSL_MLIR_PYBIND_ADAPTORS
  NAMES mlir/Bindings/Python/PybindAdaptors.h
  PATHS ${MLIR_INCLUDE_DIRS}
  NO_DEFAULT_PATH
)
if(NOT CATLASS_DSL_MLIR_PYBIND_ADAPTORS)
  message(FATAL_ERROR
          "MLIR Python PybindAdaptors.h is required for the typed Python bridge.")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${TLA_REPO_ROOT}/.artifacts/pydeps:$ENV{PYTHONPATH}"
          ${Python3_EXECUTABLE} -c
          "import pathlib; import mlir._mlir_libs as libs; root = pathlib.Path(libs.__file__).resolve().parent; candidates = [root / 'libMLIRPythonCAPI.so', *root.glob('libMLIRPythonCAPI.so.*')]; print(next(str(p) for p in candidates if p.exists()))"
  OUTPUT_VARIABLE CATLASS_DSL_MLIR_PYTHON_CAPI_LIBRARY
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(CATLASS_DSL_MLIR_PYTHON_CAPI_LIBRARY STREQUAL "")
  message(FATAL_ERROR
          "libMLIRPythonCAPI.so from the active Python MLIR package is required "
          "for the typed Python bridge.")
endif()
get_filename_component(CATLASS_DSL_MLIR_PYTHON_CAPI_DIR
                       "${CATLASS_DSL_MLIR_PYTHON_CAPI_LIBRARY}" DIRECTORY)

if(NOT TARGET LLVM)
  find_package(LLVM REQUIRED CONFIG HINTS "${MLIR_CMAKE_DIR}/../llvm")
endif()

set(METAPASSES_TBLGEN_EXE "${MLIR_TABLEGEN_EXE}")
if(NOT METAPASSES_TBLGEN_EXE)
  set(METAPASSES_TBLGEN_EXE "${MLIR_TBLGEN}")
endif()
set(METAPASSES_TBLGEN_INCLUDES
  ${MLIR_TBLGEN_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(METAPASSES_TBLGEN_INCLUDE_FLAGS)
foreach(_inc ${METAPASSES_TBLGEN_INCLUDES})
  if (_inc)
    list(APPEND METAPASSES_TBLGEN_INCLUDE_FLAGS -I ${_inc})
  endif()
endforeach()
set(METAPASSES_TBLGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblgen)

# tla dialect TableGen outputs (dialect/attrs/types/ops).
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
    ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
  COMMAND ${CMAKE_COMMAND} -E make_directory ${METAPASSES_TBLGEN_OUTPUT_DIR}/tla
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-dialect-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-dialect-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-attrdef-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-attrdef-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-enum-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-enum-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-typedef-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-typedef-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-op-defs ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
  COMMAND ${METAPASSES_TBLGEN_EXE} --gen-op-decls ${TLA_TD_FILE} ${METAPASSES_TBLGEN_INCLUDE_FLAGS} -o ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
  COMMAND_EXPAND_LISTS
  DEPENDS ${TLA_TD_FILE}
  COMMENT "Generate TableGen files for tla"
)
add_custom_target(TlaTblgen DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Dialect.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Attrs.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Enums.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Types.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.cpp.inc
  ${CMAKE_CURRENT_BINARY_DIR}/tblgen/tla/Ops.h.inc
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${CATLASS_DSL_HIVM_SOURCE_INCLUDE_DIR} ${CATLASS_DSL_HIVM_GENERATED_INCLUDE_DIR})
link_directories(${CATLASS_DSL_HIVM_LIB_DIR})
set(CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/python/catlass)
file(MAKE_DIRECTORY ${CATLASS_DSL_PYTHON_PACKAGE_OUTPUT_DIR})

add_subdirectory(lib)
add_subdirectory(tools/tla-compile)

# Convenience target: build all compiler tools.
add_custom_target(tla-compiler DEPENDS TlaCompile _tla_type_bridge_native)
if (TARGET bishengir_template_bitcode)
  add_dependencies(tla-compiler bishengir_template_bitcode)
endif()

# Lit regression tests for compiler tools.
set(CATLASS_DSL_TLA_COMPILE_TOOL
    "${CMAKE_CURRENT_BINARY_DIR}/tools/tla-compile/TlaCompile")
if (EXISTS "${TLA_REPO_ROOT}/tests/lit/CMakeLists.txt")
  add_subdirectory(${TLA_REPO_ROOT}/tests/lit ${CMAKE_CURRENT_BINARY_DIR}/tests/lit)
endif()
