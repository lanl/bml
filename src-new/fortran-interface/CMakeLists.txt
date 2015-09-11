set(FORTRAN-SOURCES
  bml.F90
  bml_allocate.F90
  bml_types.F90)
add_library(bml-fortran ${FORTRAN-SOURCES})

foreach(S ${FORTRAN-SOURCES})
  get_filename_component(MODFILE ${S} NAME_WE)
  list(APPEND FORTRAN-HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${MODFILE}.mod)
endforeach()
install(FILES ${FORTRAN-HEADERS} DESTINATION include)
