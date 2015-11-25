!> Transpose functions.
module bml_transpose_m
  use, intrinsic :: iso_c_binding
  use bml_types_m
  implicit none
  private

  interface

     function bml_transpose_new_C(a) bind(C, name="bml_transpose_new")
       import :: C_PTR
       type(C_PTR), value, intent(in) :: a
       type(C_PTR) :: bml_transpose_new_C
     end function bml_transpose_new_C

  end interface

  public :: bml_transpose

contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @param a_t The transpose.
  subroutine bml_transpose(a, a_t)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: a_t

    a_t%ptr = bml_transpose_new_C(a%ptr)

  end subroutine bml_transpose

end module bml_transpose_m
