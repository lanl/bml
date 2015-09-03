!> @copyright Los Alamos National Laboratory 2015

!> The matrix types.
module bml_type_m

  implicit none

  !> The dense matrix type name.
  !!
  !! @ingroup allocate_group
  character(len=*), parameter :: BML_MATRIX_DENSE = "dense"

  !> The sparse (ELLPACK \cite ellpack) matrix type name.
  !!
  !! @ingroup allocate_group
  character(len=*), parameter :: BML_MATRIX_ELLPACK = "ellpack"

  !> Matrix single precision.
  !!
  !! @ingroup allocate_group
  character(len=*), parameter :: BML_PRECISION_SINGLE = "single"

  !> Matrix double precision.
  !!
  !! @ingroup allocate_group
  character(len=*), parameter :: BML_PRECISION_DOUBLE = "double"

  !> The matrix type.
  !!
  !! @ingroup allocate_group
  type, abstract :: bml_matrix_t
     !> The matrix size, assumed square.
     integer :: N = -1
   contains
     !> Returns the matrix type as a string.
     !!
     !! The strings is composed of "type":"precision".
     procedure(get_type), deferred, nopass :: get_type
  end type bml_matrix_t

  !> Returns the matrix type as a string.
  !!
  !! The strings is composed of "type":"precision".
  abstract interface
     function get_type() result(type_name)
       character(len=:), pointer :: type_name
     end function get_type
  end interface

end module bml_type_m
