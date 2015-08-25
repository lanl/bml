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

set(BUILD_SHARED_LIBS FALSE
  CACHE BOOL "Whether to build shared libraries (the default is static).")

add_subdirectory(src)

include(FindDoxygen)

if(DOXYGEN_FOUND)
  configure_file(docs/Doxyfile.in Doxyfile)
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