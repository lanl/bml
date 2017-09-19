program ptr

  use, intrinsic :: iso_C_binding
  use bml

  type(bml_matrix_t) :: a
  real(C_FLOAT), allocatable :: diagonal(:)
  real(C_FLOAT), allocatable :: row(:)

  call bml_random_matrix(BML_MATRIX_ELLPACK, BML_ELEMENT_REAL, C_FLOAT, 10, 10, A, BML_DMODE_SEQUENTIAL)
  !call bml_identity_matrix(BML_MATRIX_ELLPACK, BML_ELEMENT_REAL, C_FLOAT, 10, 10, A, BML_DMODE_SEQUENTIAL)
  call bml_get_diagonal(A, diagonal)
  call bml_get_row(A, 1, row)

  call bml_print_matrix("A", A, 0, 10, 0, 10)
  write(*, *) "diagonal =", diagonal
  write(*, *) "row =", row

  call bml_deallocate(A)

end program ptr
