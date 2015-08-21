program test

  use bml
  use bml_type_dense_m
  use bml_print_dense_m

  implicit none

  integer, parameter :: N = 12

  class(bml_matrix_t), allocatable :: A
  class(bml_matrix_t), allocatable :: B
  double precision, allocatable :: A_dense(:, :)

  call random_matrix(MATRIX_TYPE_NAME_DENSE_DOUBLE, N, A)
  call convert_to_dense(A, A_dense)

  select type(A)
  type is(bml_matrix_dense_t)
     if(maxval(A_dense-A%matrix) > 1e-12) then
        call error(__FILE__, __LINE__, "element mismatch")
     endif
  class default
     call error(__FILE__, __LINE__, "error")
  end select

  call convert_from_dense(MATRIX_TYPE_NAME_DENSE_DOUBLE, A_dense, B)

  select type(B)
  type is(bml_matrix_dense_t)
     if(maxval(A_dense-B%matrix) > 1e-12) then
        call print_matrix("A", A)
        call print_matrix_dense("B", B)
        call error(__FILE__, __LINE__, "element mismatch")
     endif
  class default
     call error(__FILE__, __LINE__, "error")
  end select

end program test
