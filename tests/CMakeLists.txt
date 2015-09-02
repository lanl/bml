include_directories(${CMAKE_BINARY_DIR}/src)

# To add more tests, simply add the source to this variable. The CMake
# script will automatically build and run the test.
set(TESTS
  allocate_matrix.F90
  convert_matrix.F90)

# Generate the test_modules module.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "module test_modules\n")
foreach(FILE ${TESTS})
  get_filename_component(BASENAME ${FILE} NAME_WE)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "  use ${BASENAME}_m\n")
  message(STATUS "Adding module ${BASENAME}_m")
endforeach()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90 "end module test_modules\n")

# Scan the sources in TESTS and generate build rules to build and test
# them. All tests are assumed to be programs.
foreach(FILE ${TESTS})
  get_filename_component(DIRNAME ${FILE} DIRECTORY)
  if(DIRNAME STRGREATER "")
    message(FATAL_ERROR "Please no paths in tests")
  endif()
  get_filename_component(BASENAME ${FILE} NAME_WE)

  message(STATUS "Adding test ${BASENAME}")

  add_executable(${BASENAME}
    ${FILE}
    test_m.F90
    #${CMAKE_CURRENT_BINARY_DIR}/test_modules.F90
    test_driver.F90)
  set_target_properties(${BASENAME}
    PROPERTIES
    COMPILE_DEFINITIONS "TEST_MODULE=${BASENAME}_m;TEST_TYPE=${BASENAME}_t")
  target_link_libraries(${BASENAME} bml)
  if(BLAS_FOUND)
    target_link_libraries(${BASENAME} ${BLAS_LIBRARIES})
  endif()
  add_test(${BASENAME} ${BASENAME})
endforeach()
