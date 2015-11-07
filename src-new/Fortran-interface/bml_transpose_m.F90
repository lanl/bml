!> Transpose functions.
module bml_transpose_m

  implicit none
  private

  interface

     subroutine bml_transpose_C(a, a_t) bind(C, name="bml_transpose")
       use, intrinsic :: iso_C_binding
       type(C_PTR), value, intent(in) :: a
       type(C_PTR), value, intent(in) :: a_t
     end subroutine bml_transpose_C

  end interface

  public :: bml_transpose

contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @param a_t The transpose.
  subroutine bml_transpose(a, a_t)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: a_t

    call bml_transpose_C(a%ptr, a_t%ptr)

  end subroutine bml_transpose

end module bml_transpose_m
