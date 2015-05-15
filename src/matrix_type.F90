!> @copyright Los Alamos National Laboratory 2015

!> The matrix types.
module matrix_type

  implicit none

  !> The dense matrix type name.
  character(len=*), parameter :: matrix_type_name_dense = "dense"

  !> The matrix type.
  type :: matrix_t

     !> The particular matrix type.
     character(len=100) :: matrix_type = "undefined"

     !> The matrix size, assumed square.
     integer :: N = -1

     !> The dense matrix.
     double precision, allocatable :: dense_matrix(:, :)

  end type matrix_t

end module matrix_type
