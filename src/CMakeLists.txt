set( SOURCES
  matrix_add.F90
  matrix_add_dense.F90
  matrix_initialize.F90
  matrix_initialize_dense.F90
  matrix_multiply.F90
  matrix_multiply_dense.F90
  matrix_type.F90
  bml.F90
  )

add_library( bml ${SOURCES} )

unset( HEADERS )
foreach( S ${SOURCES} )
  get_filename_component( TEMP ${S} NAME_WE )
  list( APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${TEMP}.mod )
endforeach()

install( TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  )

install( FILES ${HEADERS}
  DESTINATION include
  )
