cmake_minimum_required( VERSION 2.8 )
project( bml Fortran )

set( PROJECT_VERSION "0.1" )
set( PROJECT_DESCRIPTION
  "A versatile matrix library for linear algebra operations."
  )

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

add_subdirectory( src )

include( FindDoxygen )

if( DOXYGEN_FOUND )
  configure_file( Doxyfile.in Doxyfile )
  add_custom_target( doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    )
endif()

enable_testing()

add_subdirectory( tests )
