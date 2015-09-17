include_directories(..)

set(SOURCES-DENSE
  bml_allocate_dense.c
  bml_convert_dense.c
  bml_introspection_dense.c)
add_library(bml-dense OBJECT ${SOURCES-DENSE})
