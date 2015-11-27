include_directories(
  ${CMAKE_SOURCE_DIR}/src/C-interface
  ${CMAKE_BINARY_DIR}/src/Fortran-interface)

set(C-TESTS
  add_matrix.c
  allocate_matrix.c
  convert_matrix.c
  copy_matrix.c
  diagonalize_matrix.c
  io_matrix.c
  multiply_matrix.c
  multiply_banded_matrix.c
  scale_matrix.c
  threshold_matrix.c
  trace_matrix.c
  transpose_matrix.c)

set(TYPES dense ellpack)

include(${CMAKE_SOURCE_DIR}/cmake/bmlGetTypes.cmake)

# Scan the sources in C-TESTS and generate build rules to build and
# test them. All tests are assumed to be functions using the
# bml_test.h test framework.
foreach(FILE ${C-TESTS})
  get_filename_component(DIRNAME ${FILE} DIRECTORY)
  if(DIRNAME STRGREATER "")
    message(FATAL_ERROR "Please no paths in tests")
  endif()
  get_filename_component(BASENAME ${FILE} NAME_WE)

  message(STATUS "Adding test ${BASENAME}")

  foreach(TYPE ${TYPES})
    set(MATRIX_TYPE bml_matrix_${TYPE}_t)
    foreach(i RANGE ${NUMBER_TYPES_MAX})
      get_C_types(${i} MATRIX_PRECISION REAL_TYPE)
      string(TOUPPER ${MATRIX_PRECISION} PRECISION_DEF)
      set(TESTNAME C-${BASENAME}-${TYPE}-${MATRIX_PRECISION})
      unset(COMPILE_DEFINITIONS)
      list(APPEND COMPILE_DEFINITIONS ${PRECISION_DEF})
      list(APPEND COMPILE_DEFINITIONS MATRIX_PRECISION=${MATRIX_PRECISION})
      list(APPEND COMPILE_DEFINITIONS MATRIX_TYPE=${MATRIX_TYPE})
      list(APPEND COMPILE_DEFINITIONS MATRIX_TYPE_NAME=${TYPE})
      list(APPEND COMPILE_DEFINITIONS REAL_T=${REAL_TYPE})
      add_executable(${TESTNAME}
        ${FILE}
        test_driver.c)
      set_target_properties(${TESTNAME}
        PROPERTIES
        COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}")
      if(OPENMP_FOUND)
        set_target_properties(${TESTNAME}
          PROPERTIES
          COMPILE_FLAGS ${OpenMP_C_FLAGS}
          LINK_FLAGS ${OpenMP_C_FLAGS})
      endif()
      target_link_libraries(${TESTNAME} bml ${LINK_LIBRARIES})
      add_test(${TESTNAME} ${TESTNAME})
      if(VALGRIND AND NOT OPENMP_FOUND)
        add_test(${TESTNAME}-valgrind
          ${VALGRIND}
          --error-exitcode=1
          --leak-check=full
          --read-var-info=yes
          --track-origins=yes
          ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME})
      endif()
    endforeach()
  endforeach()
endforeach()

# To add more tests, simply add the source to this variable. The CMake
# script will automatically build and run the test.
set(FORTRAN-TESTS
  add_matrix.F90
  allocate_matrix.F90
  bandwidth.F90
  convert_matrix.F90
  copy_matrix.F90
  diagonalize_matrix.F90
  get_bandwidth.F90
  get_matrix.F90
  io_matrix.F90
  multiply_matrix.F90
  scale_matrix.F90
  threshold_matrix.F90
  trace_matrix.F90
  transpose_matrix.F90)

# Generate the test_modules module.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "module test_modules\n")
foreach(FILE ${FORTRAN-TESTS})
  get_filename_component(BASENAME ${FILE} NAME_WE)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "  use ${BASENAME}_m\n")
  message(STATUS "Adding module ${BASENAME}_m")
endforeach()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "end module test_modules\n")

# Scan the sources in TESTS and generate build rules to build and test
# them. All tests are assumed to be procedures using the interface
# from test_m.
foreach(FILE ${FORTRAN-TESTS})
  get_filename_component(DIRNAME ${FILE} DIRECTORY)
  if(DIRNAME STRGREATER "")
    message(FATAL_ERROR "Please no paths in tests")
  endif()
  get_filename_component(BASENAME ${FILE} NAME_WE)

  message(STATUS "Adding test ${BASENAME}")

  foreach(TYPE ${TYPES})
    set(MATRIX_TYPE BML_MATRIX_${TYPE})
    foreach(i RANGE ${NUMBER_TYPES_MAX})
      get_Fortran_types(${i} MATRIX_PRECISION PRECISION_NAME REAL_TYPE REAL_KIND REAL_NAME)
      set(PRECISION BML_PRECISION_${MATRIX_PRECISION})
      set(TESTNAME ${BASENAME}-${TYPE}-${MATRIX_PRECISION})
      string(TOLOWER ${TESTNAME} TESTNAME)
      string(TOUPPER ${MATRIX_PRECISION} PRECISION_TYPE)
      set(TESTNAME Fortran-${TESTNAME})
      unset(COMPILE_DEFINITIONS)
      list(APPEND COMPILE_DEFINITIONS TEST_MODULE=${BASENAME}_m)
      list(APPEND COMPILE_DEFINITIONS TEST_TYPE=${BASENAME}_t)
      list(APPEND COMPILE_DEFINITIONS MATRIX_PRECISION="${MATRIX_PRECISION}")
      list(APPEND COMPILE_DEFINITIONS MATRIX_TYPE=${MATRIX_TYPE})
      list(APPEND COMPILE_DEFINITIONS REAL_TYPE=${REAL_TYPE})
      list(APPEND COMPILE_DEFINITIONS PRECISION_NAME=${PRECISION_NAME})
      list(APPEND COMPILE_DEFINITIONS ${PRECISION_TYPE})
      add_executable(${TESTNAME}
        ${FILE}
        test_m.F90
        test_driver.F90)
      set_target_properties(${TESTNAME}
        PROPERTIES
        COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}")
      if(OPENMP_FOUND)
        set_target_properties(${TESTNAME}
          PROPERTIES
          COMPILE_FLAGS ${OpenMP_Fortran_FLAGS}
          LINK_FLAGS ${OpenMP_Fortran_FLAGS})
      endif()
      target_link_libraries(${TESTNAME} bml ${LINK_LIBRARIES})
      add_test(${TESTNAME} ${TESTNAME})
    endforeach()
  endforeach()
endforeach()
