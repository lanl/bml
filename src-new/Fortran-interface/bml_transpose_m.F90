!> Transpose functions.
module bml_transpose_m
  implicit none
contains

  !> Return the transpose of a matrix.
  !!
  !! @param a The matrix.
  !! @param a_t The transpose.
  subroutine bml_transpose(a, a_t)

    use bml_types_m

    class(bml_matrix_t), intent(in) :: a
    class(bml_matrix_t), allocatable, intent(out) :: a_t

  end subroutine bml_transpose

end module bml_transpose_m
