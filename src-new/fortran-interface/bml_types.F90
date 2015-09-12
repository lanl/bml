!> The basic bml types.
module bml_types

  use, intrinsic :: iso_C_binding

  !> The bml matrix type.
  type :: bml_matrix_t
     type(C_PTR) :: ptr = C_NULL_PTR
  end type bml_matrix_t

  !> The bml-dense matrix type identifier.
  character(len=*), parameter :: BML_MATRIX_DENSE = "dense"

  !> The single precision identifier.
  character(len=*), parameter :: BML_PRECISION_SINGLE = "single-precision"

  !> The double-precision identifier.
  character(len=*), parameter :: BML_PRECISION_DOUBLE = "double-precision"

end module bml_types
