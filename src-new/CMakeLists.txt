set(HEADERS
  bml.h
  bml_allocate.h
  bml_convert.h
  bml_logger.h
  bml_types.h
  bml_utilities.h)
install(FILES ${HEADERS} DESTINATION include)

add_library(bml-new
  bml_allocate.c
  bml_convert.c
  bml_logger.c
  bml_utilities.c)

add_subdirectory(dense)
add_subdirectory(fortran-interface)

target_link_libraries(bml-new bml-dense bml-fortran)
install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
