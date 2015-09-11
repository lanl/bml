add_subdirectory(dense)

set(HEADERS
  bml.h
  bml_allocate.h
  bml_convert.h
  bml_logger.h
  bml_types.h
  bml_utilities.h)

add_library(bml-new
  bml_allocate.c
  bml_convert.c
  bml_logger.c
  bml_utilities.c)

target_link_libraries(bml-new bml-dense)
install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)