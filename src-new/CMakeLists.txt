add_subdirectory(dense)

add_library(bml-new
  bml_allocate.c
  bml_convert.c
  bml_logger.c)
target_link_libraries(bml-new bml-dense)