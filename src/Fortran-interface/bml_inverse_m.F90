!> Matrix inverse functions.
module bml_inverse_m

  use bml_c_interface_m
  use bml_introspection_m
  use bml_types_m

  implicit none
  private

  public :: bml_inverse

contains

  !> Matrix inverse.
  !!
  !! @param a The matrix.
  !! @param b The inverse of matrix a
  subroutine bml_inverse(a, b)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: b

    call bml_deallocate(b)
    b%ptr = bml_inverse_C(a%ptr)

  end subroutine bml_inverse

end module bml_inverse_m
