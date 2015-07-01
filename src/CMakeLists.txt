set(SOURCES
  bml_add.F90
  bml_add_dense.F90
  bml_error.F90
  bml_initialize.F90
  bml_initialize_dense.F90
  bml_multiply.F90
  bml_multiply_dense.F90
  bml_print.F90
  bml_print_dense.F90
  bml_type.F90
  bml_type_dense.F90
  bml.F90)

add_library(bml ${SOURCES})

unset(HEADERS)
foreach(S ${SOURCES})
  get_filename_component(TEMP ${S} NAME_WE)
  list(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${TEMP}.mod)
endforeach()

install(TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${HEADERS}
  DESTINATION include)
