cmake_minimum_required(VERSION 2.8.8)
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

include(CheckFunctionExists)
set(CMAKE_REQUIRED_LIBRARIES "-lm")
check_function_exists(fabs HAVE_FABS)
if(NOT HAVE_FABS)
  message(FATAL_ERROR "Could not find the fabs() function")
else()
  list(APPEND LINK_LIBRARIES "-lm")
endif()

include(FindThreads)

set(BML_OPENMP TRUE CACHE BOOL "Whether to compile with OpenMP support")
if(BML_OPENMP)
  include(FindOpenMP)
endif()

set(BML_CUDA FALSE CACHE BOOL "Whether to compile with CUDA support")
if(BML_CUDA)
  include(FindCUDA)
endif()

set(BML_OPENCL FALSE CACHE BOOL "Whether to compiler with OpenCL support")
if(BML_OPENCL)
  include(FindOpenCL)
endif()

set(BLAS_VENDOR ""
  CACHE STRING "If set, the preferred BLAS/LAPACK vendor. Possible choices: {Intel,ACML}")

if(BLAS_VENDOR STREQUAL "Intel")
  message(STATUS "Attempting to use Intel's BLAS/LAPACK (MKL)")
  set(BLA_VENDOR Intel10_64lp)
elseif(BLAS_VENDOR STREQUAL "ACML")
  message(STATUS "Attempting to use AMD's BLAS/LAPACK (ACML)")
  set(BLA_VENDOR ACML)
else()
  message(STATUS "Using first BLAS/LAPACK found")
endif()

include(FindBLAS)
include(FindLAPACK)

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "Could not find BLAS library.")
endif()

if(NOT LAPACK_FOUND)
  message(FATAL_ERROR "Could not find LAPACK library.")
endif()

macro(check_C_Fortran_function_exists FUNCTION VARIABLE)
  # Test various naming schemes for calling a Fortran function from C.
  set(${VARIABLE} ${FUNCTION}-NOTFOUND)
  foreach(FUNC ${FUNCTION} ${FUNCTION}_ ${FUNCTION}__)
    check_function_exists(${FUNC} HAVE_${FUNC})
    if(HAVE_${FUNC})
      set(${VARIABLE} ${FUNC})
      break()
    endif()
  endforeach()
  if(${ARGC} GREATER 2)
    if(${ARGV2} STREQUAL REQUIRED AND NOT ${VARIABLE})
      message(FATAL_ERROR "Can not find function ${FUNCTION}")
    endif()
  endif()
endmacro()

include(CheckFortranFunctionExists)
if(BLAS_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_fortran_function_exists(dgemm HAVE_DGEMM)
  check_fortran_function_exists(sgemm HAVE_SGEMM)
  check_C_Fortran_function_exists(dgemm C_DGEMM REQUIRED)
  check_C_Fortran_function_exists(sgemm C_SGEMM REQUIRED)
  check_C_Fortran_function_exists(sscal C_SSCAL REQUIRED)
  check_C_Fortran_function_exists(dscal C_DSCAL REQUIRED)
  add_definitions(
    -DC_SGEMM=${C_SGEMM}
    -DC_DGEMM=${C_DGEMM}
    -DC_SSCAL=${C_SSCAL}
    -DC_DSCAL=${C_DSCAL})
endif()
if(LAPACK_FOUND)
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  check_fortran_function_exists(dsyev HAVE_DSYEV)
endif()

add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
if(BLAS_FOUND)
  add_definitions(-DHAVE_BLAS)
endif()
if(LAPACK_FOUND)
  add_definitions(-DHAVE_LAPACK)
endif()
if(HAVE_DGEMM)
  add_definitions(-DHAVE_DGEMM)
endif()
if(HAVE_SGEMM)
  add_definitions(-DHAVE_SGEMM)
endif()
if(HAVE_DSYEV)
  add_definitions(-DHAVE_DSYEV)
endif()

if(BLAS_FOUND)
  list(APPEND LINK_LIBRARIES ${BLAS_LIBRARIES})
endif()
if(LAPACK_FOUND)
  list(APPEND LINK_LIBRARIES ${LAPACK_LIBRARIES})
endif()

set(INDENT_ARGS -gnu -nut -i4 -bli0 -npcs -bfda)

find_program(INDENT indent)
message(STATUS "Found indent: ${INDENT}")
mark_as_advanced(INDENT)

set(DIST_FILES CMakeLists.txt)

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
      ${CMAKE_SOURCE_DIR}/src-new/C-interface
      ${CMAKE_SOURCE_DIR}/src-new/Fortran-interface)
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
else()
  if(BML_NEW)
    add_custom_target(indent-tests)
  endif()
endif()

if(INDENT AND BML_NEW)
  message(STATUS "Creating indent target")
  add_custom_target(indent
    DEPENDS
    indent-C-interface
    indent-dense
    indent-ellpack
    indent-tests)
else()
  add_custom_target(indent)
endif()

find_program(HAVE_ETAGS etags)
mark_as_advanced(HAVE_ETAGS)
if(HAVE_ETAGS)
  add_custom_target(tags
    ${HAVE_ETAGS} ${CMAKE_SOURCE_DIR}/src/*.F90)
endif()

list(APPEND DIST_FILES
  ${CMAKE_SOURCE_DIR}/CMakeLists.txt
  ${CMAKE_SOURCE_DIR}/README.md
  ${CMAKE_SOURCE_DIR}/bml-manual.pdf)

find_program(TAR tar)
if(TAR)
  message(STATUS "Creating dist target")
  add_custom_target(dist
    ${TAR}
    -C ${CMAKE_SOURCE_DIR}
    -cvf ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}-${PROJECT_VERSION}.tar
    ${DIST_FILES})
else()
  add_custom_target(dist)
endif()