!> Transpose functions.
module bml_transpose_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_transpose

contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @param a_t The transpose.
  subroutine bml_transpose(a, a_t)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: a_t

    call bml_deallocate(a_t)
    a_t%ptr = bml_transpose_new_C(a%ptr)

  end subroutine bml_transpose

end module bml_transpose_m
