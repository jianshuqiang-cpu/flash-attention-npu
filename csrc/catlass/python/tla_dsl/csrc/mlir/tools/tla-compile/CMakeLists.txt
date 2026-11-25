find_package(Threads REQUIRED)

# Host-built LLVM needs host glibc (>=2.32). Conda clang++ links against an
# older sysroot and fails with missing __libc_single_threaded / mallinfo2.
# Drive the final link with system g++ so those symbols resolve.
if(CMAKE_CXX_COMPILER MATCHES "conda" AND EXISTS "/usr/bin/g++")
  set(CMAKE_CXX_LINK_EXECUTABLE
    "/usr/bin/g++ <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

add_executable(TlaCompile main.cpp)
target_include_directories(TlaCompile PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/tblgen
  ${MLIR_INCLUDE_DIRS}
)
# LLVMSupport needs pthread; clang+conda link does not pull it transitively.
target_link_libraries(TlaCompile PRIVATE
  TlaCompilePipeline
  TlaPasses
  MLIRParser
  Threads::Threads
)
