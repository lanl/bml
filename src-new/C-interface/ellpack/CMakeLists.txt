include_directories(..)

set(HEADERS-ELLPACK
  bml_allocate_ellpack.h
  bml_convert_ellpack.h
  bml_copy_ellpack.h
  bml_introspection_ellpack.h
  bml_scale_ellpack.h
  bml_types_ellpack.h)

set(SOURCES-ELLPACK
  bml_allocate_ellpack.c
  bml_convert_ellpack.c
  bml_copy_ellpack.c
  bml_introspection_ellpack.c
  bml_scale_ellpack.c)
add_library(bml-ellpack OBJECT ${SOURCES-ELLPACK})
set_target_properties(bml-ellpack
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

if(INDENT)
  unset(INDENT-ELLPACK)
  foreach(f ${HEADERS-ELLPACK} ${SOURCES-ELLPACK})
    list(APPEND INDENT-ELLPACK ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-ellpack
    ${INDENT} ${INDENT_ARGS} ${INDENT-ELLPACK})
else()
  add_custom_target(indent-ellpack)
endif()
