program test

  use bml
  use bml_type_dense_m
  use bml_print_dense_m
  use test_m

  implicit none

  integer, parameter :: N = 12

  call test_function(N, BML_MATRIX_DENSE, BML_PRECISION_SINGLE)
  call test_function(N, BML_MATRIX_DENSE, BML_PRECISION_DOUBLE)
  call test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_SINGLE)
  call test_function(N, BML_MATRIX_ELLPACK, BML_PRECISION_DOUBLE)

end program test
