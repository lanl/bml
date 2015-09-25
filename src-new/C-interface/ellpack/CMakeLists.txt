include_directories(..)

set(HEADERS-ELLPACK
  bml_add_ellpack.h
  bml_allocate_ellpack.h
  bml_convert_ellpack.h
  bml_copy_ellpack.h
  bml_introspection_ellpack.h
  bml_multiply_ellpack.h
  bml_scale_ellpack.h
  bml_trace_ellpack.h
  bml_types_ellpack.h)

set(SOURCES-ELLPACK
  bml_add_ellpack.c
  bml_allocate_ellpack.c
  bml_convert_ellpack.c
  bml_copy_ellpack.c
  bml_introspection_ellpack.c
  bml_multiply_ellpack.c
  bml_scale_ellpack.c
  bml_trace_ellpack.c)

add_library(bml-ellpack OBJECT ${SOURCES-ELLPACK})
set_target_properties(bml-ellpack
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

set(SOURCES-ELLPACK-TYPED
  bml_add_ellpack_typed.c
  bml_allocate_ellpack_typed.c
  bml_copy_ellpack_typed.c
  bml_multiply_ellpack_typed.c
  bml_scale_ellpack_typed.c
  bml_trace_ellpack_typed.c)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-ellpack single_real "${SOURCES-ELLPACK-TYPED}")
bml_add_typed_library(bml-ellpack double_real "${SOURCES-ELLPACK-TYPED}")
set_target_properties(bml-ellpack-single_real
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
set_target_properties(bml-ellpack-double_real
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

if(INDENT)
  message(STATUS "Setting up indent-ellpack")
  unset(INDENT-ELLPACK)
  foreach(f ${HEADERS-ELLPACK} ${SOURCES-ELLPACK} ${SOURCES-ELLPACK-TYPED})
    list(APPEND INDENT-ELLPACK ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-ellpack
    ${INDENT} ${INDENT_ARGS} ${INDENT-ELLPACK})
else()
  add_custom_target(indent-ellpack)
endif()
