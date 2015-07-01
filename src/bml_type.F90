!> @copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type

  implicit none

  !> The dense matrix type name.
  character(len=*), parameter :: MATRIX_TYPE_NAME_DENSE = "dense"

  !> The matrix type.
  type, abstract :: matrix_t

     !> The matrix size, assumed square.
     integer :: N = -1

  end type matrix_t

end module bml_type
