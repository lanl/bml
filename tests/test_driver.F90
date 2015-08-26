program test

  use bml
  use bml_type_dense_m
  use bml_print_dense_m
  use test_m

  implicit none

  interface
     function test_function(n, matrix_type, matrix_precision)
       integer, intent(in) :: n
       character(len=*), intent(in) :: matrix_type
       character(len=*), intent(in) :: matrix_precision
       logical :: test_function
     end function test_function
  end interface

  integer, parameter :: N = 129

  write(*, "(A)") "Testing "//BML_MATRIX_DENSE//":"//BML_PRECISION_SINGLE
  if(.not. test_function(N, BML_MATRIX_DENSE, BML_PRECISION_SINGLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_DENSE//":"//BML_PRECISION_DOUBLE
  if(.not. test_function(N, BML_MATRIX_DENSE, BML_PRECISION_DOUBLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_ELLPACK//":"//BML_PRECISION_SINGLE
  if(.not. test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_SINGLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

  write(*, "(A)") "Testing "//BML_MATRIX_ELLPACK//":"//BML_PRECISION_DOUBLE
  if(.not. test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_DOUBLE)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

end program test
