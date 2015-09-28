include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../C-interface)

set(FORTRAN-SOURCES-PUBLIC
  bml.F90
  bml_allocate_m.F90
  bml_add_m.F90
  bml_convert_m.F90
  bml_copy_m.F90
  bml_diagonalize_m.F90
  bml_introspection_m.F90
  bml_multiply_m.F90
  bml_scale_m.F90
  bml_trace_m.F90
  bml_transpose_m.F90
  bml_types_m.F90
  bml_utilities_m.F90)

set(FORTRAN-SOURCES-PRIVATE
  bml_interface_m.F90
  bml_print_matrix_wrapper.c
  bml_convert_from_dense_wrapper.c)

set(FORTRAN-SOURCES-TYPED
  bml_add_typed_m.F90)

set(MATRIX_TYPES single_real double_real single_complex double_complex)
list(LENGTH MATRIX_TYPES NUMBER_TYPES)
math(EXPR NUMBER_TYPES_MAX "${NUMBER_TYPES}-1")
foreach(i RANGE ${NUMBER_TYPES_MAX})
  list(GET MATRIX_TYPES ${i} MATRIX_TYPE)
  foreach(S ${FORTRAN-SOURCES-TYPED})
    string(REPLACE "typed" ${MATRIX_TYPE} S_TYPED ${S})
    get_filename_component(S_TYPED ${S_TYPED} NAME)
    add_custom_command(OUTPUT ${S_TYPED}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${S}
      COMMAND ${CMAKE_SOURCE_DIR}/scripts/convert-template
      -DMATRIX_TYPE=${MATRIX_TYPE}
      -o ${S_TYPED}
      ${CMAKE_CURRENT_SOURCE_DIR}/${S})
    list(APPEND FORTRAN-SOURCES-PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/${S_TYPED})
  endforeach()
endforeach()

add_library(bml-fortran OBJECT ${FORTRAN-SOURCES-PUBLIC} ${FORTRAN-SOURCES-PRIVATE})
if(OPENMP_FOUND)
  set_target_properties(bml-fortran
    PROPERTIES COMPILE_FLAGS ${OpenMP_C_FLAGS})
endif()

foreach(S ${FORTRAN-SOURCES-PUBLIC})
  get_filename_component(MODFILE ${S} NAME_WE)
  list(APPEND FORTRAN-HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${MODFILE}.mod)
endforeach()
install(FILES ${FORTRAN-HEADERS} DESTINATION include)
