!> \copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_dense

  use bml_type_m

  implicit none

  !> The bml dense matrix type.
  type, extends(bml_matrix_t) :: bml_matrix_dense_t

     !> The dense matrix.
     double precision, allocatable :: matrix(:, :)

  end type bml_matrix_dense_t

end module bml_type_dense
