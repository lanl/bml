include_directories(..)

set(HEADERS-DENSE
  bml_add_dense.h
  bml_allocate_dense.h
  bml_convert_dense.h
  bml_copy_dense.h
  bml_introspection_dense.h
  bml_multiply_dense.h
  bml_scale_dense.h
  bml_threshold_dense.h
  bml_trace_dense.h
  bml_types_dense.h)

set(SOURCES-DENSE
  bml_add_dense.c
  bml_allocate_dense.c
  bml_convert_dense.c
  bml_copy_dense.c
  bml_introspection_dense.c
  bml_multiply_dense.c
  bml_scale_dense.c
  bml_threshold_dense.c
  bml_trace_dense.c)

add_library(bml-dense OBJECT ${SOURCES-DENSE})
if(OPENMP_FOUND)
  set_target_properties(bml-dense
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

set(SOURCES-DENSE-TYPED
  bml_add_dense_typed.c
  bml_allocate_dense_typed.c
  bml_convert_dense_typed.c
  bml_copy_dense_typed.c
  bml_multiply_dense_typed.c
  bml_scale_dense_typed.c
  bml_threshold_dense_typed.c
  bml_trace_dense_typed.c)

include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-dense single_real "${SOURCES-DENSE-TYPED}")
bml_add_typed_library(bml-dense double_real "${SOURCES-DENSE-TYPED}")
bml_add_typed_library(bml-dense single_complex "${SOURCES-DENSE-TYPED}")
bml_add_typed_library(bml-dense double_complex "${SOURCES-DENSE-TYPED}")
if(OPENMP_FOUND)
  set_target_properties(bml-dense-single_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-dense-double_real
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-dense-single_complex
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-dense-double_complex
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()
