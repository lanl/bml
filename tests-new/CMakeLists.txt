include_directories(${CMAKE_SOURCE_DIR}/src-new)

set(TESTS
  convert.c)

# Scan the sources in TESTS and generate build rules to build and test
# them. All tests are assumed to be programs.
foreach(FILE ${TESTS})
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
      set(TESTNAME ${BASENAME}-${TYPE}-${MATRIX_PRECISION})
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
