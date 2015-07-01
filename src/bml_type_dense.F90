!> @copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_dense

  use bml_type

  implicit none

  type, extends(matrix_t) :: matrix_dense_t

     !> The dense matrix.
     double precision, allocatable :: dense_matrix(:, :)

  end type matrix_dense_t

end module bml_type_dense
