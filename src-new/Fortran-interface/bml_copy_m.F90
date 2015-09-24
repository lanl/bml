!> Copy operations on matrices.
module bml_copy_m
  implicit none
contains

  !> Copy (assign) a matrix to another one.
  !!
  !! This operation performs \f$ B \leftarrow A \f$.
  !!
  !! \param a Matrix to copy.
  !! \param b Matrix to copy to.
  subroutine bml_copy(a, b)

    use bml_types

    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: b

  end subroutine bml_copy

end module bml_copy_m
