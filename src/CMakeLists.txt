set(SOURCES
  bml.F90
  bml_add_m.F90
  bml_allocate_m.F90
  bml_convert_m.F90
  bml_error_m.F90
  bml_multiply_m.F90
  bml_print_m.F90
  bml_scale_m.F90
  bml_trace_m.F90
  bml_type_m.F90
  dense/bml_add_dense.F90
  dense/bml_allocate_dense.F90
  dense/bml_convert_dense.F90
  dense/bml_multiply_dense.F90
  dense/bml_print_dense.F90
  dense/bml_scale_dense.F90
  dense/bml_trace_dense_m.F90
  dense/bml_type_dense.F90)

set(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/bml.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_add_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_allocate_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_convert_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_error_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_multiply_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_print_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_scale_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_trace_m.mod
  ${CMAKE_CURRENT_BINARY_DIR}/bml_type_m.mod)

add_library(bml ${SOURCES})

install(TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${HEADERS}
  DESTINATION include)
