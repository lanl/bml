include_directories(
  ${CMAKE_SOURCE_DIR}/src/C-interface
  ${CMAKE_BINARY_DIR}/src/Fortran-interface)

add_executable(getter-test-C
  test_getters.c)
target_link_libraries(getter-test-C bml ${LINK_LIBRARIES})

add_executable(getter-test-Fortran
  test_getters.F90)
target_link_libraries(getter-test-Fortran bml ${LINK_LIBRARIES})
