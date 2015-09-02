program test

  use bml
  use bml_type_dense_m
  use bml_print_dense_m
  use TEST_MODULE

  implicit none

  integer, parameter :: N = 129
  type(TEST_TYPE) :: tester

  write(*, "(A)") "Testing "//BML_MATRIX_DENSE//":"//BML_PRECISION_SINGLE
  if(.not. tester%test_function(N, BML_MATRIX_DENSE, BML_PRECISION_SINGLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_DENSE//":"//BML_PRECISION_DOUBLE
  if(.not. tester%test_function(N, BML_MATRIX_DENSE, BML_PRECISION_DOUBLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_ELLPACK//":"//BML_PRECISION_SINGLE
  if(.not. tester%test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_SINGLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_ELLPACK//":"//BML_PRECISION_DOUBLE
  if(.not. tester%test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_DOUBLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

end program test
