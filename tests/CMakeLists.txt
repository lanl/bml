include_directories( ${CMAKE_BINARY_DIR}/src )

set( TESTS
  initialize_matrix_dense.F90
  add_matrix_dense.F90
  )

foreach( FILE ${TESTS} )
  get_filename_component( DIRNAME ${FILE} DIRECTORY )
  if( DIRNAME STRGREATER "" )
    message( FATAL_ERROR "Please no paths in tests" )
  endif()
  get_filename_component( BASENAME ${FILE} NAME_WE )

  message( STATUS "Adding test ${BASENAME}" )

  add_executable( ${BASENAME} ${FILE} )
  target_link_libraries( ${BASENAME} bml )
  add_test( ${BASENAME} ${BASENAME} )
endforeach()
