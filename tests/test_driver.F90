program test

  use bml
  use bml_type_dense_m
  use bml_print_dense_m
  use TEST_MODULE

  implicit none

  integer, parameter :: N = 129
  type(TEST_TYPE) :: tester

  write(*, "(A)") "Testing "//MATRIX_TYPE//":"//PRECISION
  if(.not. tester%test_function(N, MATRIX_TYPE, PRECISION)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

end program test
