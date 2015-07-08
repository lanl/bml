cmake_minimum_required(VERSION 2.8)
project(bml Fortran)

set(PROJECT_VERSION "0.1")
set(PROJECT_DESCRIPTION
  "Basic Matrix Library (bml): A versatile matrix library for linear algebra operations")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_subdirectory(src)

include(FindDoxygen)

if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile)
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile)
endif()

enable_testing()
add_subdirectory(tests)

find_program(HAVE_ETAGS etags)
if(HAVE_ETAGS)
  add_custom_target(tags
    ${HAVE_ETAGS} ${CMAKE_SOURCE_DIR}/src/*.F90)
endif()