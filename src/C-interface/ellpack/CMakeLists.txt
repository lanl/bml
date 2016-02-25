include_directories(..)

set(HEADERS-ELLPACK
  bml_add_ellpack.h
  bml_allocate_ellpack.h
  bml_convert_ellpack.h
  bml_copy_ellpack.h
  bml_elemental_ellpack.h
  bml_introspection_ellpack.h
  bml_multiply_ellpack.h
  bml_norm_ellpack.h
  bml_normalize_ellpack.h
  bml_scale_ellpack.h
  bml_setters_ellpack.h
  bml_threshold_ellpack.h
  bml_trace_ellpack.h
  bml_transpose_ellpack.h
  bml_types_ellpack.h
  bml_utilities_ellpack.h)

set(SOURCES-ELLPACK
  bml_add_ellpack.c
  bml_allocate_ellpack.c
  bml_convert_ellpack.c
  bml_copy_ellpack.c
  bml_elemental_ellpack.c
  bml_introspection_ellpack.c
  bml_multiply_ellpack.c
  bml_norm_ellpack.c
  bml_normalize_ellpack.c
  bml_scale_ellpack.c
  bml_setters_ellpack.c
  bml_threshold_ellpack.c
  bml_trace_ellpack.c
  bml_transpose_ellpack.c
  bml_utilities_ellpack.c)

add_library(bml-ellpack OBJECT ${SOURCES-ELLPACK})
if(OPENMP_FOUND)
  set_target_properties(bml-ellpack
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

set(SOURCES-ELLPACK-TYPED
  bml_add_ellpack_typed.c
  bml_allocate_ellpack_typed.c
  bml_convert_ellpack_typed.c
  bml_copy_ellpack_typed.c
  bml_multiply_ellpack_typed.c
  bml_norm_ellpack_typed.c
  bml_normalize_ellpack_typed.c
  bml_scale_ellpack_typed.c
  bml_setters_ellpack_typed.c
  bml_threshold_ellpack_typed.c
  bml_trace_ellpack_typed.c
  bml_transpose_ellpack_typed.c
  bml_utilities_ellpack_typed.c)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-ellpack single_real "${SOURCES-ELLPACK-TYPED}")
bml_add_typed_library(bml-ellpack double_real "${SOURCES-ELLPACK-TYPED}")
bml_add_typed_library(bml-ellpack single_complex "${SOURCES-ELLPACK-TYPED}")
bml_add_typed_library(bml-ellpack double_complex "${SOURCES-ELLPACK-TYPED}")
if(OPENMP_FOUND)
  set_target_properties(bml-ellpack-single_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-ellpack-double_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-ellpack-single_complex
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-ellpack-double_complex
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()
