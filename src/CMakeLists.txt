set(SOURCES
  bml.F90
  bml_add_m.F90
  bml_allocate_m.F90
  bml_convert_m.F90
  bml_copy_m.F90
  bml_diagonalize_m.F90
  bml_error_m.F90
  bml_get_m.F90
  bml_get_bandwidth_m.F90
  bml_multiply_m.F90
  bml_print_m.F90
  bml_scale_m.F90
  bml_trace_m.F90
  bml_transpose_m.F90
  bml_type_m.F90
  bml_utility_m.F90)

set(SOURCES-PREC
  bml_add_generic_m.F90
  bml_utility_generic_m.F90)

set(SOURCES-DENSE-PREC
  dense/bml_add_dense_generic_m.F90)

set(SOURCES-DENSE
  dense/bml_allocate_dense_m.F90
  dense/bml_convert_dense_m.F90
  dense/bml_copy_dense_m.F90
  dense/bml_diagonalize_dense_m.F90
  dense/bml_get_bandwidth_dense_m.F90
  dense/bml_multiply_dense_m.F90
  dense/bml_print_dense_m.F90
  dense/bml_scale_dense_m.F90
  dense/bml_trace_dense_m.F90
  dense/bml_transpose_dense_m.F90
  dense/bml_type_dense_m.F90)

set(SOURCES-ELLPACK
  ellpack/bml_allocate_ellpack_m.F90
  ellpack/bml_convert_ellpack_m.F90
  ellpack/bml_get_bandwidth_ellpack_m.F90
  ellpack/bml_type_ellpack_m.F90)

macro(convert_template SOURCES SOURCES-single SOURCES-double)
  if(DEFINED ${SOURCES-single} OR DEFINED ${SOURCES-double})
    message(WARNING "Output variables are already defined")
  else()
    foreach(S ${SOURCES})
      string(REPLACE "generic_m.F90" "single_m.F90" S_PREC ${S})
      get_filename_component(S_PREC ${S_PREC} NAME)
      add_custom_command(OUTPUT ${S_PREC}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${S}
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/convert-template
        -DBML_REAL=real
        -DBML_PRECISION_NAME=single
        -o ${S_PREC}
        ${CMAKE_CURRENT_SOURCE_DIR}/${S})
      list(APPEND ${SOURCES-single} ${CMAKE_CURRENT_BINARY_DIR}/${S_PREC})

      string(REPLACE "generic_m.F90" "double_m.F90" S_PREC ${S})
      get_filename_component(S_PREC ${S_PREC} NAME)
      add_custom_command(OUTPUT ${S_PREC}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${S}
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/convert-template
        -DBML_REAL=\"double precision\"
        -DBML_PRECISION_NAME=double
        -o ${S_PREC}
        ${CMAKE_CURRENT_SOURCE_DIR}/${S})
      list(APPEND ${SOURCES-double} ${CMAKE_CURRENT_BINARY_DIR}/${S_PREC})
    endforeach()
  endif()
endmacro()

convert_template("${SOURCES-PREC}" SOURCES-single SOURCES-double)
convert_template("${SOURCES-DENSE-PREC}" SOURCES-DENSE-single SOURCES-DENSE-double)

unset(HEADERS)
foreach(f ${SOURCES} ${SOURCES-single} ${SOURCES-double})
  get_filename_component(f_mod ${f} NAME_WE)
  list(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${f_mod}.mod)
endforeach()

add_library(bml
  ${SOURCES}
  ${SOURCES-single}
  ${SOURCES-double}
  ${SOURCES-DENSE}
  ${SOURCES-DENSE-single}
  ${SOURCES-DENSE-double}
  ${SOURCES-ELLPACK})

set_target_properties(bml
  PROPERTIES
  COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}"
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION})

install(TARGETS bml
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES ${HEADERS}
  DESTINATION include)
