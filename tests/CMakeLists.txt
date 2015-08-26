include_directories(${CMAKE_BINARY_DIR}/src)

# To add more tests, simply add the source to this variable. The CMake
# script will automatically build and run the test.
set(TESTS
  allocate_matrix.F90)

# Scan the sources in TESTS and generate build rules to build and test
# them. All tests are assumed to be programs.
foreach(FILE ${TESTS})
  get_filename_component(DIRNAME ${FILE} DIRECTORY)
  if(DIRNAME STRGREATER "")
    message(FATAL_ERROR "Please no paths in tests")
  endif()
  get_filename_component(BASENAME ${FILE} NAME_WE)

  message(STATUS "Adding test ${BASENAME}")

  add_executable(${BASENAME} ${FILE} test_m.F90 test_driver.F90)
  target_link_libraries(${BASENAME} bml)
  if(BLAS_FOUND)
    target_link_libraries(${BASENAME} ${BLAS_LIBRARIES})
  endif()
  add_test(${BASENAME} ${BASENAME})
endforeach()
