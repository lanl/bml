program test

  use bml

  implicit none

  integer, parameter :: N = 12

  type(matrix_t) :: A
  type(matrix_t) :: B
  type(matrix_t) :: C

  call random_matrix_dense(N, A)
  call identity_matrix_dense(N, B)

  call add(A, B, C)

  if(sum(abs(C%dense_matrix-(A%dense_matrix+B%dense_matrix))) > 1d-12) then
     write(*, *) "matrix mismatch"
     error stop
  endif

end program test
