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

set(SOURCES-DENSE-TYPED
  bml_allocate_dense_typed.c)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-dense single_real ${SOURCES-DENSE-TYPED})
bml_add_typed_library(bml-dense double_real ${SOURCES-DENSE-TYPED})

if(INDENT)
  message(STATUS "Setting up indent-dense")
  unset(INDENT-DENSE)
  foreach(f ${HEADERS-DENSE} ${SOURCES-DENSE} ${SOURCES-DENSE-TYPED})
    list(APPEND INDENT-DENSE ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-dense
    ${INDENT} ${INDENT_ARGS} ${INDENT-DENSE})
else()
  add_custom_target(indent-dense)
endif()
