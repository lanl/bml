include_directories(.)

# Public headers.
set(HEADERS-C
  bml.h
  bml_add.h
  bml_allocate.h
  bml_convert.h
  bml_copy.h
  bml_diagonalize.h
  bml_elemental.h
  bml_export.h
  bml_import.h
  bml_introspection.h
  bml_logger.h
  bml_multiply.h
  bml_scale.h
  bml_threshold.h
  bml_trace.h
  bml_transpose.h
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
  bml_diagonalize.c
  bml_elemental.c
  bml_export.c
  bml_import.c
  bml_introspection.c
  bml_logger.c
  bml_multiply.c
  bml_scale.c
  bml_threshold.c
  bml_trace.c
  bml_transpose.c
  bml_utilities.c)
add_library(bml-c OBJECT ${SOURCES-C})
if(OPENMP_FOUND)
  set_target_properties(bml-c
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

add_subdirectory(dense)
add_subdirectory(ellpack)
