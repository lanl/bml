set(SOURCES
  bml.F90
  bml_add.F90
  bml_allocate.F90
  bml_convert.F90
  bml_error.F90
  bml_multiply.F90
  bml_print.F90
  bml_scale.F90
  bml_type.F90
  dense/bml_add_dense.F90
  dense/bml_allocate_dense.F90
  dense/bml_convert_dense.F90
  dense/bml_multiply_dense.F90
  dense/bml_print_dense.F90
  dense/bml_scale_dense.F90
  dense/bml_type_dense.F90)

set(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/bml.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_add.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_allocate.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_convert.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_error.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_multiply.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_print.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_scale.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_type.mod)

add_library(bml ${SOURCES})

install(TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${HEADERS}
  DESTINATION include)
