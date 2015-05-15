program test

  use bml

  implicit none

  integer, parameter :: N = 12
  type(matrix_t) :: A

  call zero_matrix_dense(N, A)

end program test
