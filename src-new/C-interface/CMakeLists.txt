include_directories(.)

# Public headers.
set(HEADERS-C
  bml.h
  bml_add.h
  bml_allocate.h
  bml_convert.h
  bml_copy.h
  bml_introspection.h
  bml_logger.h
  bml_multiply.h
  bml_scale.h
  bml_trace.h
  bml_types.h
  bml_utilities.h)
install(FILES ${HEADERS-C} DESTINATION include)

# Private headers.
set(HEADERS-C-PRIVATE
  blas.h
  typed.h)

set(SOURCES-C
  bml_add.c
  bml_allocate.c
  bml_convert.c
  bml_copy.c
  bml_introspection.c
  bml_logger.c
  bml_multiply.c
  bml_scale.c
  bml_trace.c
  bml_utilities.c)
add_library(bml-c OBJECT ${SOURCES-C})
if(OPENMP_FOUND)
  set_target_properties(bml-c
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

add_subdirectory(dense)
add_subdirectory(ellpack)

if(INDENT)
  message(STATUS "Setting up indent-C-interface")
  unset(INDENT-C)
  foreach(f ${HEADERS-C} ${HEADERS-C-PRIVATE} ${SOURCES-C})
    list(APPEND INDENT-C ${CMAKE_CURRENT_SOURCE_DIR}/${f})
  endforeach()
  add_custom_target(indent-C-interface
    ${INDENT} ${INDENT_ARGS} ${INDENT-C})
else()
  add_custom_target(indent-C-interface)
endif()
