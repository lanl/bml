set(FORTRAN-SOURCES-PUBLIC
  bml.F90
  bml_allocate.F90
  bml_convert.F90
  bml_types.F90
  bml_utilities.F90)
set(FORTRAN-SOURCES-PRIVATE
  bml_interface.F90
  bml_print_matrix_wrapper.c
  bml_convert_from_dense_wrapper.c)
add_library(bml-fortran OBJECT ${FORTRAN-SOURCES-PUBLIC} ${FORTRAN-SOURCES-PRIVATE})

foreach(S ${FORTRAN-SOURCES-PUBLIC})
  get_filename_component(MODFILE ${S} NAME_WE)
  list(APPEND FORTRAN-HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${MODFILE}.mod)
endforeach()
install(FILES ${FORTRAN-HEADERS} DESTINATION include)
