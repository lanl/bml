set(HEADERS-DISTRIBUTED
  bml_allocate_distributed2d.h
  bml_export_distributed2d.h
  bml_import_distributed2d.h
  bml_convert_distributed2d.h
  bml_copy_distributed2d.h
  bml_diagonalize_distributed2d.h
  bml_scale_distributed2d.h
  bml_add_distributed2d.h
  bml_introspection_distributed2d.h
  bml_multiply_distributed2d.h
  bml_normalize_distributed2d.h
  bml_norm_distributed2d.h
  bml_getters_distributed2d.h
  bml_setters_distributed2d.h
  bml_threshold_distributed2d.h
  bml_transpose_distributed2d.h
  bml_trace_distributed2d.h
  bml_utilities_distributed2d.h )

set(SOURCES-DISTRIBUTED
  bml_allocate_distributed2d.c
  bml_export_distributed2d.c
  bml_import_distributed2d.c
  bml_convert_distributed2d.c
  bml_copy_distributed2d.c
  bml_diagonalize_distributed2d.c
  bml_scale_distributed2d.c
  bml_add_distributed2d.c
  bml_introspection_distributed2d.c
  bml_import_distributed2d.c
  bml_multiply_distributed2d.c
  bml_normalize_distributed2d.c
  bml_norm_distributed2d.c
  bml_getters_distributed2d.c
  bml_setters_distributed2d.c
  bml_threshold_distributed2d.c
  bml_transpose_distributed2d.c
  bml_trace_distributed2d.c
  bml_utilities_distributed2d.c )

add_library(bml-distributed2d OBJECT ${SOURCES-DISTRIBUTED})
set_target_properties(bml-distributed2d
  PROPERTIES
  POSITION_INDEPENDENT_CODE yes)
if(OPENMP_FOUND)
  set_target_properties(bml-distributed2d
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

set(SOURCES-DISTRIBUTED-TYPED
  bml_export_distributed2d_typed.c
  bml_import_distributed2d_typed.c
  bml_multiply_distributed2d_typed.c
  bml_normalize_distributed2d_typed.c
  bml_diagonalize_distributed2d_typed.c
  bml_getters_distributed2d_typed.c
  bml_setters_distributed2d_typed.c
  )

include(${PROJECT_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-distributed2d single_real "${SOURCES-DISTRIBUTED-TYPED}")
bml_add_typed_library(bml-distributed2d double_real "${SOURCES-DISTRIBUTED-TYPED}")
if(BML_COMPLEX)
  bml_add_typed_library(bml-distributed2d single_complex "${SOURCES-DISTRIBUTED-TYPED}")
  bml_add_typed_library(bml-distributed2d double_complex "${SOURCES-DISTRIBUTED-TYPED}")
endif()
if(OPENMP_FOUND)
  set_target_properties(bml-distributed2d-single_real
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-distributed2d-double_real
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
  if(BML_COMPLEX)
    set_target_properties(bml-distributed2d-single_complex
      PROPERTIES
      COMPILE_FLAGS ${OpenMP_C_FLAGS})
    set_target_properties(bml-distributed2d-double_complex
      PROPERTIES
      COMPILE_FLAGS ${OpenMP_C_FLAGS})
  endif()
endif()
