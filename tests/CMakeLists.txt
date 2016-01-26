add_test(dummy /bin/true)
add_executable(bml-test
  bml_test.c
  test_multiply.c)
set_target_properties(bml-test
  PROPERTIES
  LINK_FLAGS "-fprofile-arcs -lgcov")
