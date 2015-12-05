!> Copy operations on matrices.
module bml_copy_m

  use bml_c_interface_m
  use bml_types_m

  implicit none
  private

  public :: bml_copy

contains

  !> Copy a matrix - result is a new matrix.
  !!
  !! \ingroup copy_group_F
  !!
  !! \param a Matrix to copy
  !! \param b The copy
  subroutine bml_copy(a, b)

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(out) :: b

    call bml_deallocate(b)
    b%ptr = bml_copy_new_C(a%ptr)

  end subroutine bml_copy

end module bml_copy_m
