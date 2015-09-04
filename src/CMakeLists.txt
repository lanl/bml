set(SOURCES
  bml.F90
  bml_add_m.F90
  bml_allocate_m.F90
  bml_convert_m.F90
  bml_copy_m.F90
  bml_diagonalize_m.F90
  bml_error_m.F90
  bml_get_m.F90
  bml_multiply_m.F90
  bml_print_m.F90
  bml_scale_m.F90
  bml_trace_m.F90
  bml_transpose_m.F90
  bml_type_m.F90)

set(SOURCES-DENSE
  dense/bml_add_dense_m.F90
  dense/bml_allocate_dense_m.F90
  dense/bml_convert_dense_m.F90
  dense/bml_copy_dense_m.F90
  dense/bml_diagonalize_dense_m.F90
  dense/bml_multiply_dense_m.F90
  dense/bml_print_dense_m.F90
  dense/bml_scale_dense_m.F90
  dense/bml_trace_dense_m.F90
  dense/bml_transpose_dense_m.F90
  dense/bml_type_dense_m.F90)

set(SOURCES-ELLPACK
  ellpack/bml_allocate_ellpack_m.F90
  ellpack/bml_convert_ellpack_m.F90
  ellpack/bml_type_ellpack_m.F90)

add_library(bml
  ${SOURCES}
  ${SOURCES-DENSE}
  ${SOURCES-ELLPACK})

unset(HEADERS)
foreach(f ${SOURCES})
  get_filename_component(f_mod ${f} NAME_WE)
  list(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${f_mod}.mod)
endforeach()

set_target_properties(bml
  PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION})
add_definitions(-DPROJECT_VERSION="${PROJECT_VERSION}")
if(BLAS_FOUND)
  add_definitions(-DHAVE_BLAS)
endif()
if(LAPACK_FOUND)
  add_definitions(-DHAVE_LAPACK)
endif()
if(HAVE_DGEMM)
  add_definitions(-DHAVE_DGEMM)
endif()
if(HAVE_SGEMM)
  add_definitions(-DHAVE_SGEMM)
endif()
if(HAVE_DSYEV)
  add_definitions(-DHAVE_DSYEV)
endif()

install(TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${HEADERS}
  DESTINATION include)
