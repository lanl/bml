!> \copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_dense_m

  use bml_type_m

  implicit none

  !> The dense matrix type.
  type, public, extends(bml_matrix_t) :: bml_matrix_dense_t
  end type bml_matrix_dense_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_dense_t) :: bml_matrix_dense_double_t
     !> The dense matrix.
     double precision, allocatable :: matrix(:, :)
  end type bml_matrix_dense_double_t

  !> The bml dense matrix type.
  type, extends(bml_matrix_dense_t) :: bml_matrix_dense_single_t
     !> The dense matrix.
     real, allocatable :: matrix(:, :)
  end type bml_matrix_dense_single_t

end module bml_type_dense_m
