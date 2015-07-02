program test

  use bml

  implicit none

  integer, parameter :: N = 12
  class(bml_matrix_t), allocatable :: A
  double precision, allocatable :: A_dense(:, :)

  call zero_matrix(MATRIX_TYPE_NAME_DENSE, N, A)
  call convert_to_dense(A, A_dense)

  if(maxval(A_dense) /= 0 .or. minval(A_dense) /= 0) then
     call error(__FILE__, __LINE__, "incorrect zero matrix")
  endif

  call random_matrix(MATRIX_TYPE_NAME_DENSE, N, A)

  call identity_matrix(MATRIX_TYPE_NAME_DENSE, N, A)

  call deallocate_matrix(A)

end program test
