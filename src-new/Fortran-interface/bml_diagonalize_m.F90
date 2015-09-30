!> Matrix diagonalization functions.
module bml_diagonalize_m
  implicit none
contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvectors The set of eigenvectors.
  !! @param eigenvalues The corresponding eigenvalues.
  subroutine bml_diagonalize(a, eigenvectors, eigenvalues)

    use bml_types_m

    type(bml_matrix_t), intent(in) :: a
    type(bml_matrix_t), intent(inout) :: eigenvectors
    type(bml_vector_t), intent(inout) :: eigenvalues

  end subroutine bml_diagonalize

end module bml_diagonalize_m
