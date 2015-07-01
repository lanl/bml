program test

  use bml

  implicit none

  integer, parameter :: N = 12

  class(bml_matrix_t), allocatable :: A
  class(bml_matrix_t), allocatable :: B
  class(bml_matrix_t), allocatable :: C

  call random_matrix("dense", N, A)
  call identity_matrix("dense", N, B)

  call multiply(A, B, C)

  select type(A)
  type is(bml_matrix_dense_t)
     select type(B)
     type is(bml_matrix_dense_t)
        select type(C)
        type is(bml_matrix_dense_t)
           if(sum(abs(C%matrix-matmul(A%matrix, B%matrix))) > 1d-12) then
              call print_matrix_dense("A", A)
              call print_matrix_dense("B", B)
              call print_matrix_dense("C", C)
              call error(__FILE__, __LINE__, "matrix mismatch")
           endif
        end select
     end select
  end select

end program test
