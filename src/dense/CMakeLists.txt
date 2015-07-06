set(SOURCES
  bml_add_dense.F90
  bml_allocate_dense.F90
  bml_convert_dense.F90
  bml_dense.F90
  bml_multiply_dense.F90
  bml_print_dense.F90
  bml_scale_dense.F90
  bml_type_dense.F90)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
add_library(bml-dense ${SOURCES})

install(TARGETS bml-dense
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
