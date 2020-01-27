include_directories(..)

set(HEADERS-CSR
  bml_types_csr.h
  bml_allocate_csr.h
  bml_setters_csr.h
  bml_getters_csr.h
  bml_copy_csr.h
  bml_trace_csr.h
  bml_scale_csr.h
)


set(SOURCES-CSR
  bml_allocate_csr.c
  bml_setters_csr.c
  bml_getters_csr.c
  bml_copy_csr.c
  bml_trace_csr.c
  bml_scale_csr.c
)

add_library(bml-csr OBJECT ${SOURCES-CSR})
set_target_properties(bml-csr
  PROPERTIES
  POSITION_INDEPENDENT_CODE yes)
if(OPENMP_FOUND)
  set_target_properties(bml-csr
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

set(SOURCES-CSR-TYPED
  bml_allocate_csr_typed.c
  bml_setters_csr_typed.c
  bml_getters_csr_typed.c
  bml_copy_csr_typed.c
  bml_trace_csr_typed.c
  bml_scale_csr_typed.c
  )
  
include(${CMAKE_SOURCE_DIR}/cmake/bmlAddTypedLibrary.cmake)
bml_add_typed_library(bml-csr single_real "${SOURCES-CSR-TYPED}")
bml_add_typed_library(bml-csr double_real "${SOURCES-CSR-TYPED}")
bml_add_typed_library(bml-csr single_complex "${SOURCES-CSR-TYPED}")
bml_add_typed_library(bml-csr double_complex "${SOURCES-CSR-TYPED}")
if(OPENMP_FOUND)
  set_target_properties(bml-csr-single_real
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-csr-double_real
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-csr-single_complex
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
  set_target_properties(bml-csr-double_complex
    PROPERTIES
    COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()
