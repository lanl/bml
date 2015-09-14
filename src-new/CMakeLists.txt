set(HEADERS
  bml.h
  bml_allocate.h
  bml_convert.h
  bml_introspection.h
  bml_logger.h
  bml_types.h
  bml_utilities.h)
install(FILES ${HEADERS} DESTINATION include)

add_subdirectory(dense)
add_subdirectory(fortran-interface)

add_library(bml-new
  $<TARGET_OBJECTS:bml-dense>
  $<TARGET_OBJECTS:bml-fortran>
  bml_allocate.c
  bml_convert.c
  bml_introspection.c
  bml_logger.c
  bml_utilities.c)

install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
