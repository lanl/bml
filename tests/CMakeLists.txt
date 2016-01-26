add_test(dummy /bin/true)
add_executable(bml_test
  bml_test.c)
target_link_libraries(bml_test bml)
