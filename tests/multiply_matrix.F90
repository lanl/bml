program test

  use bml

  implicit none

  integer, parameter :: N = 12

  class(bml_matrix_t), allocatable :: A
  class(bml_matrix_t), allocatable :: B
  class(bml_matrix_t), allocatable :: C

  double precision, allocatable :: A_dense(:, :)
  double precision, allocatable :: B_dense(:, :)
  double precision, allocatable :: C_dense(:, :)

  call random_matrix(MATRIX_TYPE_NAME_DENSE, N, A)
  call identity_matrix(MATRIX_TYPE_NAME_DENSE, N, B)

  call convert_to_dense(A, A_dense)
  call convert_to_dense(B, B_dense)

  call multiply(A, B, C)

  call convert_to_dense(C, C_dense)

  if(maxval(matmul(A_dense, B_dense)-C_dense) > 1e-12) then
     call error(__FILE__, __LINE__, "incorrect matrix product")
  endif

  call deallocate_matrix(A)
  call deallocate_matrix(B)
  call deallocate_matrix(C)

end program test
