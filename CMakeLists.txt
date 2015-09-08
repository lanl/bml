cmake_minimum_required(VERSION 2.8)
project(bml Fortran)

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
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

list(APPEND GNU_Fortran_FLAGS_DEBUG "-O0 -g -fcheck=all")
list(APPEND GNU_Fortran_FLAGS_RELEASE "-O2 -g")
list(APPEND Intel_Fortran_FLAGS_DEBUG "-O0 -g -check all -traceback")
list(APPEND Intel_Fortran_FLAGS_RELEASE "-O2 -g")

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
  set(CMAKE_Fortran_FLAGS_${BUILD_TYPE} ${${CMAKE_Fortran_COMPILER_ID}_Fortran_FLAGS_${BUILD_TYPE}})
  string(REPLACE ";" " " "${CMAKE_Fortran_FLAGS_${BUILD_TYPE}}" CMAKE_Fortran_FLAGS_${BUILD_TYPE})
else()
  message(STATUS "Unknown Fortran compiler ${CMAKE_Fortran_COMPILER_ID}")
endif()

set(BUILD_SHARED_LIBS FALSE
  CACHE BOOL "Whether to build shared libraries (the default is static).")

include(FindBLAS)
include(FindLAPACK)

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

include(FindCUDA)

add_subdirectory(src)

include(FindDoxygen)

if(DOXYGEN_FOUND)
  configure_file(docs/Doxyfile.in Doxyfile)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile)
else()
  add_custom_target(doc)
endif()

set(BML_TESTING FALSE
  CACHE BOOL "Whether to build the test suite.")

if(BML_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

find_program(HAVE_ETAGS etags)
if(HAVE_ETAGS)
  add_custom_target(tags
    ${HAVE_ETAGS} ${CMAKE_SOURCE_DIR}/src/*.F90)
endif()