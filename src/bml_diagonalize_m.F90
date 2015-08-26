  !> Matrix diagonalization functions.
module bml_diagonalize_m
  implicit none
contains

  !> Diagonalize a matrix.
  !!
  !! @param a The matrix.
  !! @param eigenvectors The set of eigenvectors.
  !! @param eigenvalues The corresponding eigenvalues.
  subroutine diagonalize(a, eigenvectors, eigenvalues)

    use bml_type_m

    class(bml_matrix_t), intent(in) :: a
    double precision, intent(out) :: eigenvectors(:, :)
    double precision, intent(out) :: eigenvalues(:)

  end subroutine diagonalize

end module bml_diagonalize_m
