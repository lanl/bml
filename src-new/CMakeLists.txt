add_subdirectory(dense)

set(HEADERS bml_types.h)
add_library(bml-new
  bml_allocate.c
  bml_convert.c
  bml_logger.c)
target_link_libraries(bml-new bml-dense)
install(TARGETS bml-new
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)