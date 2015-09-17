set(SOURCES-ELLPACK
  bml_allocate_ellpack.c
  bml_convert_ellpack.c
  bml_copy_ellpack.c
  bml_introspection_ellpack.c)
add_library(bml-ellpack OBJECT ${SOURCES-ELLPACK})
