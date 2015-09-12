cmake_minimum_required(VERSION 2.8)
project(bml C CXX Fortran)

# The library version is versioned off the major version. If the API
# changes, the library version should be bumped.
set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "1")
set(PROJECT_VERSION_PATCH "0")

set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(PROJECT_DESCRIPTION
  "Basic Matrix Library (bml): A matrix library for linear algebra operations, supporting several data structures.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  message(STATUS "No user specified build type, using default...")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

list(APPEND GNU_C_FLAGS_DEBUG "-O0 -g -Wall -std=c99")
list(APPEND GNU_C_FLAGS_RELEASE "-O2 -g -std=c99")
list(APPEND GNU_Fortran_FLAGS_DEBUG "-O0 -g -fcheck=all -ffree-line-length-none")
list(APPEND GNU_Fortran_FLAGS_RELEASE "-O2 -g -ffree-line-length-none")
list(APPEND Intel_C_FLAGS_DEBUG "-O0 -g -std=c99 -check=conversions,stack,uninit -traceback")
list(APPEND Intel_C_FLAGS_RELEASE "-O2 -g -std=c99")
list(APPEND Intel_Fortran_FLAGS_DEBUG "-O0 -g -check all -traceback")
list(APPEND Intel_Fortran_FLAGS_RELEASE "-O2 -g")

if(CMAKE_BUILD_TYPE)
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR
      CMAKE_C_COMPILER_ID STREQUAL "Intel")
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
    set(CMAKE_C_FLAGS_${BUILD_TYPE} ${${CMAKE_C_COMPILER_ID}_C_FLAGS_${BUILD_TYPE}})
    string(REPLACE ";" " " "${CMAKE_C_FLAGS_${BUILD_TYPE}}" CMAKE_C_FLAGS_${BUILD_TYPE})
  else()
    message(STATUS "Unknown C compiler ${CMAKE_C_COMPILER_ID}")
  endif()

  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" OR
      CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
    set(CMAKE_Fortran_FLAGS_${BUILD_TYPE} ${${CMAKE_Fortran_COMPILER_ID}_Fortran_FLAGS_${BUILD_TYPE}})
    string(REPLACE ";" " " "${CMAKE_Fortran_FLAGS_${BUILD_TYPE}}" CMAKE_Fortran_FLAGS_${BUILD_TYPE})
  else()
    message(STATUS "Unknown Fortran compiler ${CMAKE_Fortran_COMPILER_ID}")
  endif()
endif()

set(BUILD_SHARED_LIBS FALSE
  CACHE BOOL "Whether to build shared libraries (the default is static).")

include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES "-lm")
check_function_exists(fabs HAVE_FABS)
if(NOT HAVE_FABS)
  message(FATAL_ERROR "Could not find the fabs() function")
else()
  list(APPEND LINK_LIBRARIES "-lm")
endif()

include(FindThreads)
include(FindBLAS)
include(FindLAPACK)

set(BML_CUDA FALSE CACHE BOOL "Whether to compile with CUDA support")
if(BML_CUDA)
  include(FindCUDA)
endif()

set(BML_OPENCL FALSE CACHE BOOL "Whether to compiler with OpenCL support")
if(BML_OPENCL)
  include(FindOpenCL)
endif()

include(CheckFortranFunctionExists)
if(BLAS_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_fortran_function_exists(dgemm HAVE_DGEMM)
  check_fortran_function_exists(sgemm HAVE_SGEMM)
endif()
if(LAPACK_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  check_fortran_function_exists(dsyev HAVE_DSYEV)
endif()

list(APPEND COMPILE_DEFINITIONS PROJECT_VERSION="${PROJECT_VERSION}")
if(BLAS_FOUND)
  list(APPEND COMPILE_DEFINITIONS HAVE_BLAS)
endif()
if(LAPACK_FOUND)
  list(APPEND COMPILE_DEFINITIONS HAVE_LAPACK)
endif()
if(HAVE_DGEMM)
  list(APPEND COMPILE_DEFINITIONS HAVE_DGEMM)
endif()
if(HAVE_SGEMM)
  list(APPEND COMPILE_DEFINITIONS HAVE_SGEMM)
endif()
if(HAVE_DSYEV)
  list(APPEND COMPILE_DEFINITIONS HAVE_DSYEV)
endif()

if(BLAS_FOUND)
  list(APPEND LINK_LIBRARIES ${BLAS_LIBRARIES})
endif()
if(LAPACK_FOUND)
  list(APPEND LINK_LIBRARIES ${LAPACK_LIBRARIES})
endif()

set(BML_NEW FALSE
  CACHE BOOL "Whether to build the new version of the library")

if(BML_NEW)
  add_subdirectory(src-new)
else()
  add_subdirectory(src)
endif()

include(FindDoxygen)

if(DOXYGEN_FOUND)
  if(BML_NEW)
    set(DOXYGEN_INPUT
      ${CMAKE_SOURCE_DIR}/src-new
      ${CMAKE_SOURCE_DIR}/src-new/fortran-interface)
  else()
    set(DOXYGEN_INPUT
      ${CMAKE_SOURCE_DIR}/src
      ${CMAKE_BINARY_DIR}/src)
  endif()
  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT}")
  configure_file(docs/Doxyfile.in Doxyfile)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile)
else()
  add_custom_target(docs)
endif()

set(BML_TESTING FALSE
  CACHE BOOL "Whether to build the test suite.")

if(BML_TESTING)
  message(STATUS "Setting up test suite.")
  find_program(VALGRIND valgrind
    DOC "valgrind - a suite of tools for debugging and profiling programs")
  enable_testing()
  if(BML_NEW)
    add_subdirectory(tests-new)
  else()
    add_subdirectory(tests)
  endif()
endif()

find_program(HAVE_ETAGS etags)
mark_as_advanced(HAVE_ETAGS)
if(HAVE_ETAGS)
  add_custom_target(tags
    ${HAVE_ETAGS} ${CMAKE_SOURCE_DIR}/src/*.F90)
endif()
