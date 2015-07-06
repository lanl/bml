program test

  use bml

  implicit none

  integer, parameter :: N = 12
  double precision, parameter :: alpha = 1.2

  class(bml_matrix_t), allocatable :: A
  class(bml_matrix_t), allocatable :: C

  double precision, allocatable :: A_dense(:, :)
  double precision, allocatable :: C_dense(:, :)

  call random_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, N, A)
  call scale(alpha, A, C)

  call convert_to_dense(A, A_dense)
  call convert_to_dense(C, C_dense)

  if(maxval(alpha*A_dense-C_dense) > 1e-12) then
     call error(__FILE__, __LINE__, "matrix element mismatch")
  endif

end program test
