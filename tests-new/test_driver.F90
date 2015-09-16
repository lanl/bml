program test

  use bml
  use TEST_MODULE

  implicit none

  integer, parameter :: N = 7, M = 7
  type(TEST_TYPE) :: tester

  write(*, "(A)") "Testing "//MATRIX_TYPE//":"//PRECISION
  if(.not. tester%test_function(N, MATRIX_TYPE, PRECISION, M)) then
     write(*, "(A)") "Test failed"
     error stop
  end if

end program test
