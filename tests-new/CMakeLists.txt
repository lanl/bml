include_directories(
  ${CMAKE_SOURCE_DIR}/src-new/C-interface
  ${CMAKE_BINARY_DIR}/src-new/Fortran-interface)

set(C-TESTS
  allocate_matrix.c
  convert_matrix.c)

# Scan the sources in C-TESTS and generate build rules to build and
# test them. All tests are assumed to be functions using the
# bml_test.h test framework.
foreach(FILE ${C-TESTS})
  get_filename_component(DIRNAME ${FILE} DIRECTORY)
  if(DIRNAME STRGREATER "")
    message(FATAL_ERROR "Please no paths in tests")
  endif()
  get_filename_component(BASENAME ${FILE} NAME_WE)

  foreach(TYPE dense)
    set(MATRIX_TYPE bml_matrix_${TYPE}_t)
    foreach(PRECISION single double)
      set(MATRIX_PRECISION ${PRECISION}_precision)
      if(PRECISION STREQUAL single)
        set(REAL_TYPE float)
      elseif(PRECISION STREQUAL double)
        set(REAL_TYPE double)
      else()
        message(FATAL_ERROR "Unknown type")
      endif()
      set(TESTNAME C-${BASENAME}-${TYPE}-${MATRIX_PRECISION})
      unset(COMPILE_DEFINITIONS)
      list(APPEND COMPILE_DEFINITIONS "MATRIX_PRECISION=${MATRIX_PRECISION}")
      list(APPEND COMPILE_DEFINITIONS "MATRIX_TYPE=${MATRIX_TYPE}")
      list(APPEND COMPILE_DEFINITIONS "MATRIX_TYPE_NAME=${TYPE}")
      #list(APPEND COMPILE_DEFINITIONS "REAL_TYPE=${REAL_TYPE}")
      add_executable(${TESTNAME}
        ${FILE}
        test_driver.c)
      set_target_properties(${TESTNAME}
        PROPERTIES
        COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}")
      target_link_libraries(${TESTNAME} bml-new ${LINK_LIBRARIES})
      add_test(${TESTNAME} ${TESTNAME})
      if(VALGRIND)
        add_test(valgrind-${TESTNAME}
          ${VALGRIND}
          --error-exitcode=1
          --leak-check=full
          --track-origins=yes
          ${CMAKE_CURRENT_BINARY_DIR}/${TESTNAME})
      endif()
    endforeach()
  endforeach()
endforeach()

# To add more tests, simply add the source to this variable. The CMake
# script will automatically build and run the test.
set(FORTRAN-TESTS
  allocate_matrix.F90
  convert_matrix.F90)

  # add_matrix.F90
  # copy_matrix.F90
  # diagonalize_matrix.F90
  # get_bandwidth.F90
  # get_matrix.F90
  # multiply_matrix.F90
  # scale_matrix.F90
  # trace_matrix.F90
  # transpose_matrix.F90

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

  foreach(type DENSE)
    set(MATRIX_TYPE BML_MATRIX_${type})
    foreach(precision SINGLE DOUBLE)
      set(PRECISION BML_PRECISION_${precision})
      set(TESTNAME ${BASENAME}-${type}-${precision})
      string(TOLOWER ${TESTNAME} TESTNAME)
      set(TESTNAME Fortran-${TESTNAME})
      unset(COMPILE_DEFINITIONS)
      list(APPEND COMPILE_DEFINITIONS "TEST_MODULE=${BASENAME}_m")
      list(APPEND COMPILE_DEFINITIONS "TEST_TYPE=${BASENAME}_t")
      list(APPEND COMPILE_DEFINITIONS "PRECISION=${PRECISION}")
      list(APPEND COMPILE_DEFINITIONS "MATRIX_TYPE=${MATRIX_TYPE}")
      add_executable(${TESTNAME}
        ${FILE}
        test_m.F90
        test_driver.F90)
      set_target_properties(${TESTNAME}
        PROPERTIES
        COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS}")
      target_link_libraries(${TESTNAME} bml-new ${LINK_LIBRARIES})
      add_test(${TESTNAME} ${TESTNAME})
    endforeach()
  endforeach()
endforeach()
