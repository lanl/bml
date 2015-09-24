include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../C-interface)

set(FORTRAN-SOURCES-PUBLIC
  bml.F90
  bml_allocate.F90
  bml_add_m.F90
  bml_convert.F90
  bml_copy_m.F90
  bml_diagonalize_m.F90
  bml_introspection.F90
  bml_multiply_m.F90
  bml_scale_m.F90
  bml_trace_m.F90
  bml_transpose_m.F90
  bml_types.F90
  bml_utilities.F90)

set(FORTRAN-SOURCES-PRIVATE
  bml_interface.F90
  bml_print_matrix_wrapper.c
  bml_convert_from_dense_wrapper.c)
add_library(bml-fortran OBJECT ${FORTRAN-SOURCES-PUBLIC} ${FORTRAN-SOURCES-PRIVATE})
set_target_properties(bml-fortran
  PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})

foreach(S ${FORTRAN-SOURCES-PUBLIC})
  get_filename_component(MODFILE ${S} NAME_WE)
  list(APPEND FORTRAN-HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${MODFILE}.mod)
endforeach()
install(FILES ${FORTRAN-HEADERS} DESTINATION include)
