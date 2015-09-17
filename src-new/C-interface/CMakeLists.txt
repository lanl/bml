include_directories(.)

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

if(INDENT)
  unset(INDENT-C)
  foreach(f ${HEADERS-C} ${SOURCES-C})
    list(APPEND INDENT-C ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-C-interface
    ${INDENT} ${INDENT_ARGS} ${INDENT-C})
else()
  add_custom_target(indent-C-interface)
endif()
