include_directories(..)

set(HEADERS-DENSE
  bml_allocate_dense.h
  bml_convert_dense.h
  bml_copy_dense.h
  bml_introspection_dense.h
  bml_scale_dense.h
  bml_types_dense.h)

set(SOURCES-DENSE
  bml_allocate_dense.c
  bml_convert_dense.c
  bml_copy_dense.c
  bml_introspection_dense.c
  bml_scale_dense.c)
add_library(bml-dense OBJECT ${SOURCES-DENSE})
set_target_properties(bml-dense
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

if(INDENT)
  unset(INDENT-DENSE)
  foreach(f ${HEADERS-DENSE} ${SOURCES-DENSE})
    list(APPEND INDENT-DENSE ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-dense
    ${INDENT} ${INDENT_ARGS} ${INDENT-DENSE})
else()
  add_custom_target(indent-dense)
endif()
