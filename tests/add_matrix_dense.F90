program test

  use bml

  implicit none

  integer, parameter :: N = 12

  class(matrix_t), allocatable :: A
  class(matrix_t), allocatable :: B
  class(matrix_t), allocatable :: C

  call random_matrix("dense", N, A)
  call identity_matrix("dense", N, B)

  call add(A, B, C)

  select type(A)
  type is(matrix_dense_t)
     select type(B)
     type is(matrix_dense_t)
        select type(C)
        type is(matrix_dense_t)
           if(sum(abs(C%dense_matrix-(A%dense_matrix+B%dense_matrix))) > 1d-12) then
              call error(__FILE__, __LINE__, "matrix mismatch")
           endif
        end select
     end select
  end select

end program test
