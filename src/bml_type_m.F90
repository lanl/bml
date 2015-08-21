!> \copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_m

  implicit none

  !> The single-precision dense matrix type name.
  !!
  !! \ingroup allocate_group
  !!
  !! \bug This type is not yet implemented.
  character(len=*), parameter :: MATRIX_TYPE_NAME_DENSE_SINGLE = "dense-single"

  !> The double-precision dense matrix type name.
  !!
  !! \ingroup allocate_group
  character(len=*), parameter :: MATRIX_TYPE_NAME_DENSE_DOUBLE = "dense-double"

  !> The matrix type.
  !!
  !! \ingroup allocate_group
  type, abstract :: bml_matrix_t

     !> The matrix size, assumed square.
     integer :: N = -1

  end type bml_matrix_t

end module bml_type_m
