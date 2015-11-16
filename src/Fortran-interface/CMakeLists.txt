include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../C-interface)

set(FORTRAN-SOURCES-PUBLIC
  bml.F90
  bml_add_m.F90
  bml_allocate_m.F90
  bml_convert_m.F90
  bml_copy_m.F90
  bml_diagonalize_m.F90
  bml_error_m.F90
  bml_introspection_m.F90
  bml_multiply_m.F90
  bml_scale_m.F90
  bml_threshold_m.F90
  bml_trace_m.F90
  bml_transpose_m.F90
  bml_types_m.F90
  bml_utilities_m.F90)

set(FORTRAN-SOURCES-PRIVATE
  bml_interface_m.F90
  bml_typed_m.F90)

set(FORTRAN-SOURCES-TYPED
  bml_utilities_typed_m.F90)

include(${CMAKE_SOURCE_DIR}/cmake/bmlGetTypes.cmake)
foreach(i RANGE ${NUMBER_TYPES_MAX})
  get_Fortran_types(${i} MATRIX_TYPE PRECISION_NAME REAL_TYPE)
  foreach(S ${FORTRAN-SOURCES-TYPED})
    string(REPLACE "typed" ${MATRIX_TYPE} S_TYPED ${S})
    get_filename_component(S_TYPED ${S_TYPED} NAME)
    add_custom_command(OUTPUT ${S_TYPED}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${S}
      COMMAND ${CMAKE_SOURCE_DIR}/scripts/convert-template
      -DMATRIX_TYPE=${MATRIX_TYPE}
      -DREAL_TYPE="${REAL_TYPE}"
      -DPRECISION_NAME=${PRECISION_NAME}
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
