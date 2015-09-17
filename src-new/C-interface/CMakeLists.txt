set(HEADERS-C
  bml.h
  bml_allocate.h
  bml_convert.h
  bml_copy.h
  bml_introspection.h
  bml_logger.h
  bml_types.h
  bml_utilities.h)
install(FILES ${HEADERS-C} DESTINATION include)

set(SOURCES-C
  bml_allocate.c
  bml_convert.c
  bml_copy.c
  bml_introspection.c
  bml_logger.c
  bml_utilities.c)
add_library(bml-c OBJECT ${SOURCES-C})

add_subdirectory(dense)
add_subdirectory(ellpack)
