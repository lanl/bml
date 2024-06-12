cmake_minimum_required(VERSION 3.10)

message(STATUS "CMake version ${CMAKE_VERSION}")

set(LANGUAGES C CXX)
set(BML_BUILD_FORTRAN_INTERFACE TRUE CACHE BOOL
  "Build the Fortran API (requires a Fortran compiler)")
if(BML_BUILD_FORTRAN_INTERFACE)
  list(APPEND LANGUAGES Fortran)
endif()

set(BML_BUILD_PYTHON_INTERFACE TRUE CACHE BOOL
  "Build the Python API (requires Python dev package)")

project(bml ${LANGUAGES})

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
