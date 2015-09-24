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

    use bml_types

    class(bml_matrix_t), intent(in) :: a
    double precision, allocatable, intent(out) :: eigenvectors(:, :)
    double precision, allocatable, intent(out) :: eigenvalues(:)

  end subroutine bml_diagonalize

end module bml_diagonalize_m
