program test

  use bml

  implicit none

  integer, parameter :: N = 12
  double precision, parameter :: ALPHA = 1.2
  double precision, parameter :: BETA = 0.8

  class(bml_matrix_t), allocatable :: A
  class(bml_matrix_t), allocatable :: B
  class(bml_matrix_t), allocatable :: C

  double precision, allocatable :: A_dense(:, :)
  double precision, allocatable :: B_dense(:, :)
  double precision, allocatable :: C_dense(:, :)

  integer :: i

  call random_matrix(BML_MATRIX_DENSE, N, A)
  call identity_matrix(BML_MATRIX_DENSE, N, B)

  call convert_to_dense(A, A_dense)
  call convert_to_dense(B, B_dense)

  call add(A, B, C)
  call convert_to_dense(C, C_dense)

  if(maxval(A_dense+B_dense-C_dense) > 1e-12) then
     call error(__FILE__, __LINE__, "incorrect matrix sum")
  endif

  call add_identity(A, C, alpha, beta)
  call convert_to_dense(C, C_dense)

  B_dense = alpha*A_dense
  do i = 1, N
     B_dense(i, i) = B_dense(i, i)+beta
  end do

  if(maxval(B_dense-C_dense) > 1e-12) then
     call error(__FILE__, __LINE__, "incorrect matrix add identity")
  end if

  call deallocate_matrix(A)
  call deallocate_matrix(B)
  call deallocate_matrix(C)

end program test
